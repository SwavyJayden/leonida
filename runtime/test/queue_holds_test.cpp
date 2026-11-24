// CPU-only unit tests for QueueHolds re-entrancy, sibling instances, exception safety, and cross-thread detach.
#include "../src/queue_holds.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using Holds = leonida::QueueHolds<int>;

struct MockHolder final : Holds::Holder {
    std::function<bool()> on_release;
    std::function<void(int)> on_discard;
    std::function<std::uint64_t()> on_oldest;

    bool release_ready() override { return on_release ? on_release() : false; }
    void discard_waiting_on(int sem) override {
        if (on_discard) on_discard(sem);
    }
    std::uint64_t oldest_held_seq() override { return on_oldest ? on_oldest() : 0; }
};

bool test_sibling_instance_pumps_and_removes_fence() {
    Holds a, b;
    MockHolder hb;
    bool op1_held = true;
    bool op2_held = true;
    bool op1_released = false;
    bool op2_discarded = false;
    bool b_after_ran = false;

    b.attach(&hb);
    b.add_fence(1, 0);
    b.add_fence(2, 0);
    b.held_changed(2);

    const std::uint64_t seq1 = b.next_seq();
    const std::uint64_t seq2 = b.next_seq();

    hb.on_release = [&]() {
        if (op1_held && !b.pending(1, 1)) {
            op1_held = false;
            op1_released = true;
            b.held_changed(-1);
            return true;
        }
        return false;
    };
    hb.on_discard = [&](int sem) {
        if (op2_held && sem == 2) {
            op2_held = false;
            op2_discarded = true;
            b.held_changed(-1);
        }
    };
    hb.on_oldest = [&]() -> std::uint64_t {
        if (op1_held) return seq1;
        if (op2_held) return seq2;
        return 0;
    };

    const int owner_b = 1;
    b.after_held(&owner_b, [&]() { b_after_ran = true; });
    if (b_after_ran) return false;

    const int owner_a = 2;
    bool a_after_ran = false;
    a.after_held(&owner_a, [&]() {
        a_after_ran = true;
        b.signaled(1, 1);
        b.pump_ready();
        b.remove_fence(2);
    });

    if (!a_after_ran || !op1_released || !op2_discarded || !b_after_ran) {
        std::fprintf(stderr, "sibling_instance failed: a=%d r=%d d=%d b=%d\n",
                     a_after_ran, op1_released, op2_discarded, b_after_ran);
        return false;
    }
    b.detach(&hb);
    return true;
}

bool test_exception_does_not_wedge_pump() {
    Holds q;
    const int owner = 1;
    bool caught = false;
    try {
        q.after_held(&owner, []() { throw std::runtime_error("boom"); });
    } catch (const std::runtime_error&) {
        caught = true;
    }
    if (!caught) return false;

    bool second_ran = false;
    q.after_held(&owner, [&]() { second_ran = true; });
    if (!second_ran) {
        std::fprintf(stderr, "exception wedged subsequent pump on same thread\n");
        return false;
    }
    return true;
}

bool test_reentrant_cancel_after_does_not_deadlock() {
    Holds q;
    MockHolder h;
    bool held = true;
    q.attach(&h);
    q.add_fence(1, 0);
    q.held_changed(1);
    const std::uint64_t op_seq = q.next_seq();

    h.on_release = [&]() {
        if (held && !q.pending(1, 1)) {
            held = false;
            q.held_changed(-1);
            return true;
        }
        return false;
    };
    h.on_oldest = [&]() -> std::uint64_t { return held ? op_seq : 0; };

    const int owner = 1;
    int first_count = 0;
    int second_count = 0;
    q.after_held(&owner, [&]() {
        ++first_count;
        q.cancel_after(&owner);
    });
    q.after_held(&owner, [&]() { ++second_count; });

    q.signaled(1, 1);
    q.pump_ready();
    q.detach(&h);

    if (first_count != 1 || second_count != 0) {
        std::fprintf(stderr, "reentrant cancel_after counts unexpected: first=%d second=%d\n",
                     first_count, second_count);
        return false;
    }
    return true;
}

bool test_reentrant_detach_and_snapshot_safety() {
    // Part A: re-entrant detach inside callback unblocks an already-queued barrier cleanly.
    {
        Holds q;
        MockHolder blocker;
        bool blocker_attached = true;
        q.attach(&blocker);
        blocker.on_oldest = [&]() -> std::uint64_t { return blocker_attached ? 1 : 0; };

        const int owner_free = 1;
        const int owner_wait = 2;
        bool barrier_ran = false;
        // Register barrier_free before blocker has any held ops (actually blocker reports seq 1 >= current_seq 0? No:
        // oldest_held_seq <= front.seq blocks front! So keep seq=0 first, raise inside first callback, or block only second.)
        // First holder only blocks second barrier if we call after_held with current_seq >= 1.
        (void)q.next_seq();
        // Actually run a re-entrant detach while pump_mutex_ is held during an unblocked barrier:
        blocker_attached = false;
        q.after_held(&owner_free, [&]() { q.detach(&blocker); });
        blocker_attached = true;  // if still attached, it would claim seq 1 and block owner_wait
        q.after_held(&owner_wait, [&]() { barrier_ran = true; });
        if (!barrier_ran) {
            std::fprintf(stderr, "reentrant detach failed to keep owner_wait unblocked\n");
            return false;
        }
    }

    // Part B: detaching sibling holder h2 inside h1.release_ready() must prevent calling h2->release_ready().
    {
        Holds q;
        MockHolder h1, h2;
        bool held = true;
        bool h2_destroyed = false;
        bool called_destroyed_h2 = false;

        q.attach(&h1);
        q.attach(&h2);
        q.add_fence(1, 0);
        q.held_changed(1);
        const std::uint64_t seq = q.next_seq();

        h1.on_release = [&]() {
            if (held && !q.pending(1, 1)) {
                held = false;
                q.held_changed(-1);
                q.detach(&h2);
                h2_destroyed = true;
                return true;
            }
            return false;
        };
        h1.on_oldest = [&]() -> std::uint64_t { return held ? seq : 0; };
        h2.on_release = [&]() {
            if (h2_destroyed) called_destroyed_h2 = true;
            return false;
        };

        q.signaled(1, 1);
        q.pump_ready();
        q.detach(&h1);

        if (!h2_destroyed || called_destroyed_h2) {
            std::fprintf(stderr, "snapshot safety failed: h2_destroyed=%d called_destroyed_h2=%d\n",
                         h2_destroyed, called_destroyed_h2);
            return false;
        }
    }
    return true;
}

bool test_cross_thread_detach_waits_for_running_pump() {
    Holds q;
    MockHolder h;
    q.attach(&h);

    std::mutex m;
    std::condition_variable cv;
    bool in_callback = false;
    bool allow_exit = false;
    std::atomic<bool> detached{false};

    const int owner = 1;
    std::thread t1([&]() {
        q.after_held(&owner, [&]() {
            std::unique_lock lk(m);
            in_callback = true;
            cv.notify_all();
            cv.wait(lk, [&]() { return allow_exit; });
        });
    });

    {
        std::unique_lock lk(m);
        cv.wait(lk, [&]() { return in_callback; });
    }

    std::thread t2([&]() {
        q.detach(&h);
        detached.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const bool detached_too_early = detached.load(std::memory_order_acquire);

    {
        std::lock_guard lk(m);
        allow_exit = true;
        cv.notify_all();
    }

    t2.join();
    t1.join();

    if (detached_too_early || !detached.load(std::memory_order_acquire)) {
        std::fprintf(stderr, "cross-thread detach synchronization failed: too_early=%d detached=%d\n",
                     detached_too_early, detached.load());
        return false;
    }
    return true;
}

}  // namespace

int main() {
    if (!test_sibling_instance_pumps_and_removes_fence()) return 1;
    if (!test_exception_does_not_wedge_pump()) return 2;
    if (!test_reentrant_cancel_after_does_not_deadlock()) return 3;
    if (!test_reentrant_detach_and_snapshot_safety()) return 4;
    if (!test_cross_thread_detach_waits_for_running_pump()) return 5;
    std::puts("queue_holds tests passed (5/5)");
    return 0;
}
