import numpy as np
from scipy.spatial import distance


def lmk_NEES(
    lmk_hat: np.ndarray, P_hat_lmk: np.ndarray, lmk_gt: np.ndarray
) -> float:
    '''
    Calculates NEES of landmarks. Estimated lmks are assigned to nearest true lmk, using 
    Mahalanobis distance. So multiple estimates may be assigned to the same true one.
    Reassociates estimates and true lmks each time as estimates may move over time.
    Is perhaps not needed, if so one only needs to perform association on new landmarks.
    '''
    
    assert lmk_hat.size == lmk_hat.shape[0]
    assert lmk_hat.size % 2 == 0
    assert lmk_hat.shape * 2 == P_hat_lmk.shape
    
    num_lmk_gt = lmk_gt.shape[0]
    num_lmk_hat = lmk_hat.size // 2
    lmk_hat = lmk_hat.reshape(num_lmk_hat,2)
    dists = np.zeros((num_lmk_gt,1))
    error = np.zeros_like(lmk_hat)

    
    # Find the closest true landmark to each estimate using 
    # Mahalonobis distance and assume they are associated
    # i.e. choose associations s.t. individual lmk NEES is minimized
    for i in range (num_lmk_hat):
        inds = slice(2*i, 2*i + 2)
        Pi_inv = np.linalg.inv(P_hat_lmk[inds,inds])
        
        for k in range (num_lmk_gt):
            dists[k] = distance.mahalanobis(lmk_hat[i], lmk_gt[k], Pi_inv)
        
        idx_closest = (np.abs(dists - 2)).argmin()
        error[i] = lmk_hat[i] - lmk_gt[idx_closest]
    
   
    error = error.ravel()
    NEES = error @ (np.linalg.solve(P_hat_lmk, error))
    
    return NEES

def get_gps_nis(
    pos_hat: np.ndarray, P_hat: np.ndarray, 
    pos_gps: np.ndarray, R_gps: np.ndarray, 
    sensorOffset: np.ndarray
) -> float:
    
    assert pos_hat.size == 3
    assert pos_hat.shape * 2 == P_hat.shape
    assert pos_gps.size == 2
    assert pos_gps.shape * 2 == R_gps.shape  
    
    innovation = pos_hat[:2] - pos_gps
    a = pos_hat[2]
    drotmat_dangle = - np.array([[ np.sin(a), np.cos(a)],
                                 [-np.cos(a), np.sin(a)]])
    H = np.eye(2,3)
    H[:,2] = drotmat_dangle @ sensorOffset
    S = H @ P_hat @ H.T + R_gps
    
    NIS = innovation @ (np.linalg.solve(S, innovation))
    return NIS