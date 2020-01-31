import py360convert.e2c as e2c
from PIL.Image import fromarray as I
import numpy as np


def cube_pad(equirectangular, size=224, mode='bilinear', padsize=1):
    # getting 6 faces
    datadict = e2c(equirectangular, face_w=224, mode=mode, cube_format='dict')
    # Getting cube faces
    L = datadict['L']
    R = np.flipud(np.rot90(datadict['R'], -2))
    U = np.flipud(datadict['U'])
    D = datadict['D']
    F = datadict['F']
    B = np.fliplr(datadict['B'])

    # zero padd on corner
    zeros = np.zeros((padsize, padsize, F.shape[-1])).astype(F.dtype)

    # U and D are not required to be padded
    U_ = I(U)
    D_ = I(D)

    # slices
    lu = slice(0, padsize)  # left or up
    rd = slice(-(padsize + 1), -1)

    # Padding Left face
    usplitl = np.rot90(U[:, lu, :], 1)
    dsplitl = np.rot90(D[:, lu, :], -1)
    L_ = np.concatenate((usplitl, L, dsplitl), 0)

    # Padding Front face
    lsplitr = L[:, rd, :]
    rsplitl = R[:, lu, :]
    usplitd = U[rd, :]
    dsplitu = D[lu, :]
    front_right_slice = np.concatenate((zeros, rsplitl, zeros), 0)
    front_left_slice = np.concatenate((zeros, lsplitr, zeros), 0)
    F_with_updownslice = np.concatenate((usplitd, F, dsplitu), 0)
    F_ = np.concatenate((front_left_slice, F_with_updownslice, front_right_slice), 1)

    # Padding Right face
    usplitr = np.rot90(U[:, rd, :], -1)
    dsplitr = np.rot90(D[:, rd, :], 1)
    R_ = np.concatenate((usplitr, R, dsplitr), 0)

    # Padding Back face
    lsplitl = L[:, lu, :]
    rsplitr = R[:, rd, :]
    usplitu = np.concatenate((zeros, np.fliplr(np.flipud(U[0:padsize, :])), zeros), 1)
    dsplitd = np.concatenate((zeros, np.fliplr(np.flipud(D[rd, :])), zeros), 1)
    B_with_leftrightslice = np.concatenate((rsplitr, B, lsplitl), 1)
    B_ = np.concatenate((usplitu, B_with_leftrightslice, dsplitd), 0)

    return {'F': I(F_).resize(U_.size),
            'B': I(B_).resize(U_.size),
            'L': I(L_).resize(U_.size),
            'R': I(R_).resize(U_.size),
            'U': U_,
            'D': D_, }
