# %% Imports
from plotting import ellipse
from EKFSLAM import EKFSLAM
from typing import List, Optional

from scipy.io import loadmat
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import chi2
import utils
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    print("install tqdm to have progress bar")

    # def tqdm as dummy as it is not available
    def tqdm(*args, **kwargs):
        return args[0]

# %% plot config check and style setup


# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
if "inline" in matplotlib.get_backend():
    print("Plotting is set to inline at the moment:", end=" ")

    if "ipykernel" in matplotlib.get_backend():
        print("backend is ipykernel (IPython?)")
        print("Trying to set backend to separate window:", end=" ")
        import IPython

        IPython.get_ipython().run_line_magic("matplotlib", "")
    else:
        print("unknown inline backend")

print("continuing with this plotting backend", end="\n\n\n")


# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )


def main():
    # %% Load data
    datafile = Path(__file__).parents[1].joinpath("data/simulatedSLAM")
    simSLAM_ws = loadmat(str(datafile))

    # NB: this is a MATLAB cell, so needs to "double index" to get out the measurements of a time step k:
    #
    # ex:
    #
    # z_k = z[k][0] # z_k is a (2, m_k) matrix with columns equal to the measurements of time step k
    #
    ##
    z = [zk.T for zk in simSLAM_ws["z"].ravel()]

    landmarks = simSLAM_ws["landmarks"].T
    odometry = simSLAM_ws["odometry"].T
    poseGT = simSLAM_ws["poseGT"].T

    K = len(z)
    # K = 230
    M = len(landmarks)  # TODO What to use this for?
    
    divergingRun = False
    
    # Initial params
    # Q = np.diag([0.1, 0.1, 1 * np.pi / 180]) ** 2 
    # R = np.diag([0.1, 1 * np.pi / 180]) ** 2 
    # JCBBalphas = np.array([0.001, 0.0001]) 
    
    # Bad map, good tracking
    # Q = np.diag([0.1, 0.1, 1 * np.pi / 180]) ** 2 
    # R = np.diag([1, 5 * np.pi / 180]) ** 2 
    # JCBBalphas = np.array([0.00001, 0.0000001])

    # Good params
    Q = np.diag([0.09, 0.09, 0.6 * np.pi / 180]) ** 2 
    R = np.diag([0.08, 0.8 * np.pi / 180]) ** 2 
    JCBBalphas = np.array([1e-4, 2e-6])     # to remove the 2 extra landmarks, use [1e-4, 2e-6]
    
    # Diverging
    # Q = np.diag([0.1, 0.1, 0.1 * np.pi / 180]) ** 2 
    # R = np.diag([0.1, 1 * np.pi / 180]) ** 2 
    # JCBBalphas = np.array([0.00001, 0.000001]) 
    # divergingRun = True

    doAsso = True 
    calculate_map_NEES = True
    
    doAssoPlot = False
    playMovie = False

# these can have a large effect on runtime either through the number of landmarks created
# or by the size of the association search space.

    slam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas)

    # allocate
    eta_pred: List[Optional[np.ndarray]] = [None] * K
    P_pred: List[Optional[np.ndarray]] = [None] * K
    eta_hat: List[Optional[np.ndarray]] = [None] * K
    P_hat: List[Optional[np.ndarray]] = [None] * K
    a: List[Optional[np.ndarray]] = [None] * K
    NIS = np.zeros(K)
    NISnorm = np.zeros(K)
    CI = np.zeros((K, 2))
    CInorm = np.zeros((K, 2))
    NEESes = np.zeros((K, 3))
    NEES_map = np.zeros((K,1))
    NEESnorm_map = np.zeros((K,1))
    CI_NEES_map = np.zeros((K,2))
    CInorm_NEES_map = np.zeros((K,2))
    num_assos = np.zeros((K,1), dtype=int)
    num_lmks = np.zeros((K,), dtype=int)

    # For consistency testing
    alpha = 0.95

    # init
    eta_pred[0] = poseGT[0]  # we start at the correct position for reference
    # we also say that we are 100% sure about that
    P_pred[0] = np.zeros((3, 3))

    # %% Set up plotting
    # plotting
    
    plot_folder = Path(__file__).parents[1].joinpath('plots')
    plot_folder.mkdir(exist_ok=True)
    
    if doAssoPlot:
        figAsso, axAsso = plt.subplots(num=1, clear=True)

    # %% Run simulation
    N = K

    print("starting sim (" + str(N) + " iterations)")

    for k, z_k in tqdm(enumerate(z[:N]), total=N):
        # See top: need to do "double indexing" to get z at time step k
        # Transpose is to stack measurements rowwise
        # z_k = z[k][0].T

        eta_hat[k], P_hat[k], NIS[k], a[k] = slam.update(eta_pred[k], P_pred[k], z_k)

        if k < K - 1:
            eta_pred[k + 1], P_pred[k + 1] = slam.predict(eta_hat[k], P_hat[k], odometry[k])

        assert (
            eta_hat[k].shape[0] == P_hat[k].shape[0]
        ), "dimensions of mean and covariance do not match"

        num_asso = np.count_nonzero(a[k] > -1)
        num_assos[k] = num_asso

        CI[k] = chi2.interval(alpha, 2 * num_asso)

        if num_asso > 0:
            NISnorm[k] = NIS[k] / (num_asso)
            CInorm[k] = CI[k] / (num_asso)
        else:
            NISnorm[k] = 1
            CInorm[k].fill(1)

        NEESes[k] = slam.NEESes(eta_hat[k][:3], P_hat[k][:3,:3], poseGT[k]) # TODO why not look at nees for landmarks?

        num_lmks[k] = (len(eta_hat[k]) - 3) // 2
        if num_lmks[k] > 0 and calculate_map_NEES: # Not needed in this case, but looks good
            NEES_map[k] = lmk_NEES(eta_hat[k][3:], P_hat[k][3:,3:], landmarks)
            NEESnorm_map[k] = NEES_map[k] / num_lmks[k]
            CI_NEES_map[k] = chi2.interval(alpha, 2 * num_lmks[k])
            CInorm_NEES_map[k] = CI_NEES_map[k] / num_lmks[k]
        
        if doAssoPlot and k > 0:
            axAsso.clear()
            axAsso.grid()
            zpred = slam.h(eta_pred[k]).reshape(-1, 2)
            axAsso.scatter(z_k[:, 0], z_k[:, 1], label="z")
            axAsso.scatter(zpred[:, 0], zpred[:, 1], label="zpred")
            xcoords = np.block(
                [[z_k[a[k] > -1, 0]], [zpred[a[k][a[k] > -1], 0]]]).T
            ycoords = np.block(
                [[z_k[a[k] > -1, 1]], [zpred[a[k][a[k] > -1], 1]]]).T
            for x, y in zip(xcoords, ycoords):
                axAsso.plot(x, y, lw=3, c="r")
            axAsso.legend()
            axAsso.set_title(
                f"k = {k}, {np.count_nonzero(a[k] > -1)} associations")
            plt.draw()
            plt.pause(0.001)

    print("sim complete")

    pose_est = np.array([x[:3] for x in eta_hat[:N]])
    lmk_est = [eta_hat_k[3:].reshape(-1, 2) for eta_hat_k in eta_hat]
    lmk_est_final = lmk_est[N - 1]
    
    print("Number of final landmark estimates: ", num_lmks[-1])

    np.set_printoptions(precision=4, linewidth=100)

    # %% Plotting of results
    mins = np.amin(landmarks, axis=0)
    maxs = np.amax(landmarks, axis=0)

    ranges = maxs - mins
    offsets = ranges * 0.2

    mins -= offsets
    maxs += offsets

    fig2, ax2 = plt.subplots(num=2, clear=True, figsize=(7, 5))
    # landmarks
    ax2.scatter(*landmarks.T, c="r", marker="^")
    ax2.scatter(*lmk_est_final.T, c="b", marker=".")
    # Draw covariance ellipsis of measurements
    for l, lmk_l in enumerate(lmk_est_final):
        idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
        rI = P_hat[N - 1][idxs, idxs]
        el = ellipse(lmk_l, rI, 5, 200)
        ax2.plot(*el.T, "b")

    ax2.plot(*poseGT.T[:2,:N], c="r", label="gt")
    ax2.plot(*pose_est.T[:2], c="g", label="est")
    ax2.plot(*ellipse(pose_est[-1, :2], P_hat[N - 1][:2, :2], 5, 200).T, c="g")
    ax2.set(title=f"Tracking results\n landmark estimates: {num_lmks[-1]}", xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
    ax2.axis("equal")
    ax2.grid()
    ax2.set_ylabel('[m]')
    ax2.set_xlabel('[m]')
    ax2.legend()
    fig2.canvas.manager.set_window_title("Tracking results")
    fig2.savefig(plot_folder.joinpath("Tracking results.pdf"))

    # %% Consistency

    # NIS
    insideCI = (CInorm[:N, 0] <= NISnorm[:N]) * (NISnorm[:N] <= CInorm[:N, 1])
    dof = num_assos.sum()
    ANIS = NIS.sum() / dof
    ANIS_CI = np.array(chi2.interval(alpha, 2 * dof)) / dof

    fig3, ax3 = plt.subplots(num=3, clear=True, figsize=(7, 5))
    ax3.plot(CInorm[:N, 0], '--')
    ax3.plot(CInorm[:N, 1], '--')
    ax3.plot(NISnorm[:N], lw=0.5)

    ax3.set_title(f'NIS\n {round(insideCI.mean()*100,1)}% inside CI,    ANIS: {round(ANIS,2)}, CI: {ANIS_CI.round(2)}')
    fig3.canvas.manager.set_window_title("NIS")
    fig3.savefig(plot_folder.joinpath("NIS.pdf"))

    # NEES

    fig4, ax4 = plt.subplots(nrows=3, ncols=1, figsize=(
        7, 5), num=4, clear=True, sharex=True)
    tags = ['All', 'Pos', 'Heading']
    dfs = [3, 2, 1]

    for ax, tag, NEES, df in zip(ax4, tags, NEESes.T, dfs):
        CI_NEES = chi2.interval(alpha, df)
        ax.plot(np.full(N, CI_NEES[0]), '--')
        ax.plot(np.full(N, CI_NEES[1]), '--')
        ax.plot(NEES[:N], lw=0.5)
        insideCI = (CI_NEES[0] <= NEES) * (NEES <= CI_NEES[1])

        CI_ANEES = np.array(chi2.interval(alpha, df*N)) / N
        # print(f"CI ANEES {tag}: {CI_ANEES}")
        # print(f"ANEES {tag}: {NEES.mean()}")
        
        if df == 3:
            ax.set_title(f'Robot state NEES\n {tag}:   {round(insideCI.mean()*100,1)}% inside CI,   ANEES: {round(NEES.mean(),2)}, CI: {CI_ANEES.round(2)}')
        else:
            ax.set_title(f'{tag}:   {round(insideCI.mean()*100,1)}% inside CI,   ANEES: {round(NEES.mean(),2)}, CI: {CI_ANEES.round(2)}')

    fig4.tight_layout()
    fig4.canvas.manager.set_window_title("NEES")
    fig4.savefig(plot_folder.joinpath("NEES.pdf"))

    # %% RMSE

    ylabels = ['m', 'deg']
    scalings = np.array([1, 180/np.pi])

    fig5, ax5 = plt.subplots(nrows=2, ncols=1, figsize=(
        7, 5), num=5, clear=True, sharex=True)

    pos_err = np.linalg.norm(pose_est[:N, :2] - poseGT[:N, :2], axis=1)
    heading_err = np.abs(utils.wrapToPi(pose_est[:N, 2] - poseGT[:N, 2]))

    errs = np.vstack((pos_err, heading_err))

    for ax, err, tag, ylabel, scaling in zip(ax5, errs, tags[1:], ylabels, scalings):
        ax.plot(err*scaling, label="error")
        ax.set_title(
            f"{tag}: RMSE {round(np.sqrt((err**2).mean())*scaling,3)} {ylabel}")
        ax.set_ylabel(f"[{ylabel}]")
        ax.grid()
        if tag == "Pos":
            ax_lmk = ax.twinx()
            ax_lmk.set_ylabel("Number of landmarks")
            ax_lmk.plot(num_lmks, c="indianred", label="num landmarks")
            ax_lmk.legend()

    fig5.tight_layout()
    fig5.canvas.manager.set_window_title("RMSE")
    fig5.savefig(plot_folder.joinpath("RMSE.pdf"))
    
    # Landmark NEES
    if calculate_map_NEES:
        insideCI = (CInorm_NEES_map[:N, 0] <= NEESnorm_map[:N]) * (NEESnorm_map[:N] <= CInorm_NEES_map[:N, 1])
        # TODO is this wrong?
        dof = num_lmks.sum()
        ANEES_map = NEES_map.sum() / dof
        ANEES_CI_map = np.array(chi2.interval(alpha, 2 * dof)) / dof

        fig6, ax6 = plt.subplots(num=6, clear=True, figsize=(7, 5))
        ax6.plot(CInorm_NEES_map[:N, 0], '--')
        ax6.plot(CInorm_NEES_map[:N, 1], '--')
        ax6.plot(NEESnorm_map[:N], lw=0.5)
        if not divergingRun:
            ax6.set_ylim((-0.1,5))

        ax6.set_title(f'Map NEES\n   {round(insideCI.mean()*100,1)}% inside CI,   ANEES: {round(ANEES_map,2)}, CI: {ANEES_CI_map.round(2)}')
        fig6.canvas.manager.set_window_title("Map NEES")
        fig6.savefig(plot_folder.joinpath("Map NEES.pdf"))

    # %% Movie time

    if playMovie:
        try:
            print("recording movie...")

            from celluloid import Camera

            pauseTime = 0.05
            fig_movie, ax_movie = plt.subplots(num=6, clear=True)

            camera = Camera(fig_movie)

            ax_movie.grid()
            ax_movie.set(xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
            camera.snap()

            for k in tqdm(range(N)):
                ax_movie.scatter(*landmarks.T, c="r", marker="^")
                ax_movie.plot(*poseGT[:k, :2].T, "r-")
                ax_movie.plot(*pose_est[:k, :2].T, "g-")
                ax_movie.scatter(*lmk_est[k].T, c="b", marker=".")

                if k > 0:
                    el = ellipse(pose_est[k, :2], P_hat[k][:2, :2], 5, 200)
                    ax_movie.plot(*el.T, "g")

                numLmk = lmk_est[k].shape[0]
                for l, lmk_l in enumerate(lmk_est[k]):
                    idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
                    rI = P_hat[k][idxs, idxs]
                    el = ellipse(lmk_l, rI, 5, 200)
                    ax_movie.plot(*el.T, "b")

                camera.snap()
            animation = camera.animate(interval=100, blit=True, repeat=False)
            print("playing movie")

        except ImportError:
            print(
                "Install celluloid module, \n\n$ pip install celluloid\n\nto get fancy animation of EKFSLAM."
            )

    plt.show()
    # %%

def lmk_NEES(
    lmk_hat: np.ndarray, P_hat_lmk: np.ndarray, lmk_gt: np.ndarray
) -> float:
    '''
    Calculates NEES of landmarks. Estimated lmks are assigned to nearest true lmk.
    So multiple estimates may be assigned to the same true one.
    Reassociates estimates and true lmks each time as estimates may move over time.
    Is perhaps not needed, if so one only needs to perform association on new landmarks.
    '''
    
    assert lmk_hat.size == lmk_hat.shape[0]
    assert lmk_hat.size % 2 == 0
    assert lmk_hat.shape * 2 == P_hat_lmk.shape
    
    num_lmk_gt = lmk_gt.shape[0]
    num_lmk_hat = lmk_hat.size // 2
    lmk_hat = lmk_hat.reshape(num_lmk_hat,2)
    error = np.zeros_like(lmk_hat)
    
    # Find the closest true landmark to each estimate and assume they are associated
    # TODO consider if this is wierd to do, especially when multiple estimates are assigned to same true lmk
    for i in range (num_lmk_hat):
        diff = lmk_gt - np.tile(lmk_hat[i], (num_lmk_gt,1))
        norms = np.linalg.norm(diff, 2, 1)
        error[i] = lmk_hat[i] - lmk_gt[norms.argmin()]
    
    error = error.ravel()
    NEES = error @ (np.linalg.solve(P_hat_lmk, error))
    
    return NEES

if __name__ == "__main__":
    main()
