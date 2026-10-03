import torch

def plucker_raymap_torch(c2w, K, height, width,
                         normalize_d=True,
                         squeeze_batch=False,
                         eps=1e-12):
    """
    Compute Plücker coordinates (d, m) for image rays using PyTorch only.

    Args
    ----
    c2w : torch.Tensor, shape (4,4) or (N,4,4)
        Camera-to-world transforms. If assume_c2w==False, they are treated as w2c and inverted.
    K : torch.Tensor, shape (3,3) or (N,3,3)
        Intrinsics matrix (fx, 0, cx; 0, fy, cy; 0,0,1) or batched.
    height, width : int
        Image size in pixels (H, W).
    normalize_d : bool
        Normalize ray directions to unit length (default True).
    assume_c2w : bool
        If False, treat `c2w` as w2c and invert it.
    squeeze_batch : bool
        If True and input was a single camera, return (H, W, 6) instead of (1, H, W, 6).
    eps : float
        Small epsilon to avoid division by zero.

    Returns
    -------
    rays_plucker : torch.Tensor, shape (N, H, W, 6) (or (H, W, 6) if squeezed)
        Last-dim: [d_x, d_y, d_z, m_x, m_y, m_z], where m = p x d and p is camera center in world coords.
    """

    if not torch.is_tensor(c2w):
        c2w = torch.tensor(c2w)
    if not torch.is_tensor(K):
        K = torch.tensor(K)

    # convert to float and put on same device
    device = c2w.device
    dtype = torch.float32
    c2w = c2w.to(dtype=dtype, device=device)
    K = K.to(dtype=dtype, device=device)

    # ensure batch dims
    if c2w.ndim == 2:
        c2w = c2w.unsqueeze(0)  # (1,4,4)
    if K.ndim == 2:
        K = K.unsqueeze(0)      # (1,3,3)

    N = c2w.shape[0]

    # broadcast K if needed
    if K.shape[0] == 1 and N > 1:
        K = K.expand(N, -1, -1)


    # origins and rotations
    origins = c2w[:, :3, 3]    # (N,3)
    Rs = c2w[:, :3, :3]        # (N,3,3)

    # build pixel grid (u=x, v=y) with origin at top-left, u->right, v->down
    # Using indexing='ij' for (H,W) shaped outputs
    grid_y, grid_x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                                    torch.arange(width,  device=device, dtype=dtype),
                                    indexing='ij')  # both (H,W)
    ones = torch.ones_like(grid_x)
    # pixel coordinates (u, v, 1)
    pix = torch.stack([grid_x, grid_y, ones], dim=-1)  # (H, W, 3)
    H, W = height, width
    HW = H * W

    # flatten to (3, HW)
    pix_flat = pix.reshape(-1, 3).transpose(0, 1)  # (3, HW)
    pix_flat = pix_flat.to(dtype=dtype, device=device)

    # inverse intrinsics per camera (N,3,3)
    invK = torch.linalg.inv(K)  # (N,3,3)

    # compute camera-space directions: d_cam = invK @ pix_flat, compute for all cameras at once
    # pix_flat unsqueezed to (1,3,HW) will broadcast with (N,3,3)
    pix_flat_exp = pix_flat.unsqueeze(0)  # (1,3,HW)
    # batch matmul: (N,3,3) @ (1,3,HW) -> (N,3,HW) by broadcasting
    d_cam = torch.matmul(invK, pix_flat_exp)  # (N,3,HW)

    # normalize directions if requested
    if normalize_d:
        norms = torch.linalg.norm(d_cam, dim=1, keepdim=True)  # (N,1,HW)
        d_cam = d_cam / (norms + eps)
    # rotate to world: d_world = R @ d_cam  -> (N,3,HW)
    d_world = torch.matmul(Rs, d_cam)

    # compute moment: m = p x d  where p is origin in world coords
    p = origins.unsqueeze(-1)  # (N,3,1)
    # broadcast p to (N,3,HW) and compute cross product along vector dim=1
    p_rep = p.expand(-1, -1, HW)   # (N,3,HW)
    m_world = torch.cross(p_rep, d_world, dim=1)  # (N,3,HW)

    # reshape to (N, H, W, 3)
    d_world = d_world.permute(0, 2, 1).reshape(N, H, W, 3)
    m_world = m_world.permute(0, 2, 1).reshape(N, H, W, 3)

    # concat -> (N, H, W, 6)
    rays_plucker = torch.cat([d_world, m_world], dim=-1)

    if squeeze_batch and rays_plucker.shape[0] == 1:
        return rays_plucker[0]

    return rays_plucker


def plucker_rays_paired(
    c2w_src, # [B, 1, 4, 4] c2w format, all rays computed wrt c2w_src
    c2ws, # :[B,N_views_tgt,4,4] C2W format
    intrinsics, # :[B,N_views,3,3] C2W format
    target_size=[72, 72], #h,w
    real_size=[72,72] #h,w
):
    """
    Calculates Plucker rays for 2nd view only!
    using relative pose between 1st and 2nd views
    """

    # rescale intrinsics fx,cx,fy,cy
    intrinsics[:,:,0,0] *= real_size[1]
    intrinsics[:,:,0,2] *= real_size[1]
    intrinsics[:,:,1,1] *= real_size[0]
    intrinsics[:,:,1,2] *= real_size[0]

    # calc relpose between tgt and src
    c2w_rel = torch.matmul(torch.linalg.inv(c2w_src), c2ws)
    # print(f"c2w_rel.shape: {c2w_rel.shape}")
    B,N_tgt = c2w_rel.shape[:2]

    c2w_rel = c2w_rel.reshape(B*N_tgt, *c2w_rel.shape[2:])
    intrinsics = intrinsics.reshape(B*N_tgt, *intrinsics.shape[2:])

    plucker = plucker_raymap_torch(c2w_rel,
                         intrinsics,
                         height=target_size[0],
                         width=target_size[1])
    plucker = plucker.reshape(B,N_tgt, *plucker.shape[1:])
    return plucker
