// lifecycle: console application lifecycle shims for mapped titles.
//
// Console titles never receive a normal desktop WM_CLOSE; quitting suspends the title (running its
// RegisterAppStateChangeNotification callbacks with quiesced = TRUE, giving it ~1 s to flush saves) and then ends the
// process. Additionally, console titles often size their top-level window to a 4K render target (3840x2160), which
// overflows smaller desktop work areas.
//
// This shim provides:
//   - RegisterAppStateChangeNotification / UnregisterAppStateChangeNotification on api-ms-win-core-psm-appnotify-*
//     and kernelbase.dll;
//   - CreateWindowExW / CreateWindowExA on user32.dll and ext-ms-win-rtcore-ntuser-window-*, fitting oversized
//     top-level windows into the primary monitor work area and subclassing them so SC_CLOSE runs registered suspend
//     callbacks before exiting cleanly.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "lifecycle.h"
#include "shims.h"

namespace rt::lifecycle {
namespace {

using AppStateRoutine = void(WINAPI*)(BOOLEAN quiesced, PVOID context);

struct Registration {
    std::uintptr_t id;
    AppStateRoutine routine;
    void* context;
};

constexpr wchar_t kTitleProcProp[] = L"leonida.title.wndproc";
constexpr auto kSuspendDeadline = std::chrono::seconds(1);

std::mutex& reg_mutex() {
    // Intentionally immortal so shutdown/WM_SYSCOMMAND cannot race static destruction.
    static auto* m = new std::mutex;
    return *m;
}

std::vector<Registration>& reg_list() {
    static auto* list = new std::vector<Registration>;
    return *list;
}

std::atomic<std::uintptr_t> g_next_reg_id{1};
std::atomic<bool> g_ending{false};

std::vector<Registration> snapshot_registrations() {
    std::lock_guard lock(reg_mutex());
    return reg_list();
}

void trigger_suspend_and_exit() {
    if (g_ending.exchange(true, std::memory_order_seq_cst)) return;
    auto callbacks = snapshot_registrations();
    std::fprintf(stderr, "[rt] lifecycle: window closed; running %zu suspend routine(s)\n", callbacks.size());

    // Run suspend callbacks off the UI thread so callbacks that synchronize with the window thread do not deadlock.
    std::thread([callbacks = std::move(callbacks)]() {
        for (const auto& entry : callbacks) {
            if (entry.routine) entry.routine(TRUE, entry.context);
        }
        std::fprintf(stderr, "[rt] lifecycle: title suspended; exiting\n");
        TerminateProcess(GetCurrentProcess(), 0);
    }).detach();

    std::thread([]() {
        std::this_thread::sleep_for(kSuspendDeadline);
        std::fprintf(stderr, "[rt] lifecycle: suspend deadline elapsed; exiting\n");
        TerminateProcess(GetCurrentProcess(), 0);
    }).detach();
}

LRESULT CALLBACK lifecycle_wnd_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
    if (message == WM_SYSCOMMAND && (wparam & 0xFFF0u) == SC_CLOSE) {
        trigger_suspend_and_exit();
        return 0;
    }
    const HANDLE prop = GetPropW(hwnd, kTitleProcProp);
    if (message == WM_NCDESTROY) {
        RemovePropW(hwnd, kTitleProcProp);
    }
    auto original = reinterpret_cast<WNDPROC>(prop);
    if (!original) return DefWindowProcW(hwnd, message, wparam, lparam);
    return CallWindowProcW(original, hwnd, message, wparam, lparam);
}

void install_window_hook(HWND hwnd) {
    if (!hwnd) return;
    const LONG_PTR prev = GetWindowLongPtrW(hwnd, GWLP_WNDPROC);
    if (!prev || reinterpret_cast<WNDPROC>(prev) == &lifecycle_wnd_proc) return;
    SetPropW(hwnd, kTitleProcProp, reinterpret_cast<HANDLE>(prev));
    SetWindowLongPtrW(hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&lifecycle_wnd_proc));
}

void fit_top_level_window(HWND parent, int& x, int& y, int& width, int& height) {
    if (parent != nullptr) return;
    RECT area{};
    if (!SystemParametersInfoW(SPI_GETWORKAREA, 0, &area, 0)) return;
    const auto [fitted_w, fitted_h] = fit_window_to_area(area, width, height);
    if (fitted_w == width && fitted_h == height) return;
    std::fprintf(stderr, "[rt] lifecycle: fitted oversized title window %dx%d -> %dx%d\n", width, height, fitted_w,
                 fitted_h);
    width = fitted_w;
    height = fitted_h;
    x = area.left + ((area.right - area.left) - width) / 2;
    y = area.top + ((area.bottom - area.top) - height) / 2;
}

ULONG WINAPI register_app_state_change(AppStateRoutine routine, PVOID context, PVOID* registration) {
    if (!routine || !registration) return ERROR_INVALID_PARAMETER;
    const std::uintptr_t id = g_next_reg_id.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard lock(reg_mutex());
        reg_list().push_back({id, routine, context});
    }
    *registration = reinterpret_cast<PVOID>(id);
    return ERROR_SUCCESS;
}

void WINAPI unregister_app_state_change(PVOID registration) {
    const auto id = reinterpret_cast<std::uintptr_t>(registration);
    std::lock_guard lock(reg_mutex());
    auto& list = reg_list();
    list.erase(std::remove_if(list.begin(), list.end(), [id](const Registration& r) { return r.id == id; }),
               list.end());
}

HWND WINAPI create_window_ex_w_shim(DWORD ex_style, LPCWSTR class_name, LPCWSTR window_name, DWORD style, int x, int y,
                                    int width, int height, HWND parent, HMENU menu, HINSTANCE instance, LPVOID param) {
    fit_top_level_window(parent, x, y, width, height);
    const HWND hwnd =
        ::CreateWindowExW(ex_style, class_name, window_name, style, x, y, width, height, parent, menu, instance, param);
    if (hwnd && !parent) install_window_hook(hwnd);
    return hwnd;
}

HWND WINAPI create_window_ex_a_shim(DWORD ex_style, LPCSTR class_name, LPCSTR window_name, DWORD style, int x, int y,
                                    int width, int height, HWND parent, HMENU menu, HINSTANCE instance, LPVOID param) {
    fit_top_level_window(parent, x, y, width, height);
    const HWND hwnd =
        ::CreateWindowExA(ex_style, class_name, window_name, style, x, y, width, height, parent, menu, instance, param);
    if (hwnd && !parent) install_window_hook(hwnd);
    return hwnd;
}

}  // namespace

std::pair<int, int> fit_window_to_area(const RECT& area, int width, int height) {
    const int area_w = area.right - area.left;
    const int area_h = area.bottom - area.top;
    if (width <= 0 || height <= 0 || area_w <= 0 || area_h <= 0 || (width <= area_w && height <= area_h)) {
        return {width, height};
    }
    const double scale =
        std::min(static_cast<double>(area_w) / static_cast<double>(width),
                 static_cast<double>(area_h) / static_cast<double>(height));
    return {static_cast<int>(static_cast<double>(width) * scale),
            static_cast<int>(static_cast<double>(height) * scale)};
}

std::size_t registered_notification_count() {
    std::lock_guard lock(reg_mutex());
    return reg_list().size();
}

void invoke_suspend_callbacks_for_test() {
    for (const auto& entry : snapshot_registrations()) {
        if (entry.routine) entry.routine(TRUE, entry.context);
    }
}

}  // namespace rt::lifecycle

namespace {
using namespace rt::lifecycle;

#define APP_NOTIFY_API(dll)                                                                          \
    {dll, "RegisterAppStateChangeNotification", reinterpret_cast<void*>(&register_app_state_change)}, \
    {dll, "UnregisterAppStateChangeNotification", reinterpret_cast<void*>(&unregister_app_state_change)}

#define WINDOW_API(dll)                                                                \
    {dll, "CreateWindowExW", reinterpret_cast<void*>(&create_window_ex_w_shim)},       \
    {dll, "CreateWindowExA", reinterpret_cast<void*>(&create_window_ex_a_shim)}

}  // namespace

SHIM_TABLE(lifecycle,
    APP_NOTIFY_API("api-ms-win-core-psm-appnotify-l1-1-0.dll"),
    APP_NOTIFY_API("api-ms-win-core-psm-appnotify-l1-1-1.dll"),
    APP_NOTIFY_API("kernelbase.dll"),
    WINDOW_API("user32.dll"),
    WINDOW_API("ext-ms-win-rtcore-ntuser-window-l1-1-0.dll"),
    WINDOW_API("ext-ms-win-rtcore-ntuser-window-l1-1-1.dll"),
);
