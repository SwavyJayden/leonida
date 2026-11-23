; A 3D Dref sample cannot be represented in Vulkan (VUID-StandaloneSpirv-OpImage-04777).
v_mov_b32_e32 v4, 1.0
v_mov_b32_e32 v5, 0.5
v_mov_b32_e32 v6, 0.25
v_mov_b32_e32 v7, 0.75
image_sample_c_lz v9, v[4:7], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_3D
exp mrt0 v9, v9, v9, v9 done vm
s_endpgm
