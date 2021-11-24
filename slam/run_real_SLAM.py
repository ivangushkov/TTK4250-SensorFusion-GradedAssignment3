# %% Imports
from scipy.io import loadmat
from scipy.stats import chi2
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    print("install tqdm for progress bar")

    # def tqdm as dummy
    def tqdm(*args, **kwargs):
        return args[0]


import numpy as np
from EKFSLAM import EKFSLAM
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from plotting import ellipse
from vp_utils import detectTrees, odometry, Car
from utils import rotmat2d

# %% plot config check and style setup

plot_folder = Path(__file__).parents[1].joinpath('plots')
plot_folder.mkdir(exist_ok=True)

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
    victoria_park_foler = Path(
        __file__).parents[1].joinpath("data/victoria_park")
    realSLAM_ws = {
        **loadmat(str(victoria_park_foler.joinpath("aa3_dr"))),
        **loadmat(str(victoria_park_foler.joinpath("aa3_lsr2"))),
        **loadmat(str(victoria_park_foler.joinpath("aa3_gpsx"))),
    }

    timeOdo = (realSLAM_ws["time"] / 1000).ravel()
    timeLsr = (realSLAM_ws["TLsr"] / 1000).ravel()
    timeGps = (realSLAM_ws["timeGps"] / 1000).ravel()

    steering = realSLAM_ws["steering"].ravel()
    speed = realSLAM_ws["speed"].ravel()
    LASER = (
        realSLAM_ws["LASER"] / 100
    )  # Divide by 100 to be compatible with Python implementation of detectTrees
    La_m = realSLAM_ws["La_m"].ravel()
    Lo_m = realSLAM_ws["Lo_m"].ravel()
    gps = np.vstack((Lo_m, La_m)).T

    K = timeOdo.size
    mK = timeLsr.size
    Kgps = timeGps.size

    # %% Parameters

    L = 2.83  # axel distance
    H = 0.76  # center to wheel encoder
    a = 0.95  # laser distance in front of first axel
    b = 0.5  # laser distance to the left of center

    car = Car(L, H, a, b)
    CorrCoeff = np.array([[1, 0, 0], [0, 1, 0.9], [0, 0.9, 1]])
    
    # Initial
    # sigmas = 0.025 * np.array([1e-4, 5e-5, 6 * np.pi / 180])
    # XR = np.diag([0.1, 1 * np.pi / 180]) ** 2  
    # XJCBBalphas = np.array([1e-5, 1e-6])

    # Good track
    # sigmas = 0.025 * np.array([1e-3, 1e-4, 7 * np.pi / 180])
    # R = np.diag([1, 2 * np.pi / 180]) ** 2  
    # JCBBalphas = np.array([1e-5, 1e-6])
    
    # sigmas = 0.025 * np.array([1e-3, 1e-4, 7 * np.pi / 180])
    # R = np.diag([0.1, 1 * np.pi / 180]) ** 2  
    # JCBBalphas = np.array([1e-5, 1e-6])
    
    # Test Rgps og P0 2, P0 heading også 2
    
    # Diverging with low NIS and many landmarks, run N = K//4 feks
    sigmas = 0.025 * np.array([1e-3, 1e-4, 1 * np.pi / 180])
    R = np.diag([1, 2 * np.pi / 180]) ** 2  
    JCBBalphas = np.array([5e-4, 5e-5])
    
    R_gps = np.diag([3, 3]) ** 2
    # P = np.zeros((3, 3))
    stds = np.diag([3, 3, 1 * np.pi / 180])
    P = stds @ CorrCoeff @ stds
    
    
    
    Q = np.diag(sigmas) @ CorrCoeff @ np.diag(sigmas)
    # Initialize state
    # you might want to tweak these for a good reference
    eta = np.array([Lo_m[0], La_m[1], 36 * np.pi / 180])
    
    doPlot = False
    do_raw_prediction = False
    showPlots = False

    sensorOffset = np.array([car.a + car.L, car.b])
    doAsso = True

    slam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas,
                   sensor_offset=sensorOffset)

    # For consistency testing
    alpha = 0.05
    confidence_prob = 1 - alpha

    xupd = np.zeros((mK, 3))
    a = [None] * mK
    NIS = np.zeros(mK)
    NISnorm = np.zeros(mK)
    CI = np.zeros((mK, 2))
    CInorm = np.zeros((mK, 2))
    num_assos = np.zeros((mK,1), dtype=int)
    gps_nis = np.zeros((Kgps,1))
    pos_err = np.zeros((Kgps,1))
    gps_ind = 0

    xupd[0] = eta

    mk_first = 1  # first seems to be a bit off in timing
    mk = mk_first
    t = timeOdo[0]

    # %%  run
    N = K // 4

    lh_pose = None

    if doPlot:
        fig, ax = plt.subplots(num=1, clear=True)

        lh_pose = ax.plot(eta[0], eta[1], "k", lw=3)[0]
        sh_lmk = ax.scatter(np.nan, np.nan, c="r", marker="x")
        sh_Z = ax.scatter(np.nan, np.nan, c="b", marker=".")

    if do_raw_prediction:
        odos = np.zeros((K, 3))
        odox = np.zeros((K, 3))
        odox[0] = eta
        P_odo = P.copy()
        for k in range(min(N, K - 1)):
            odos[k + 1] = odometry(speed[k + 1], steering[k + 1], 0.025, car)
            odox[k + 1], _ = slam.predict(odox[k], P_odo, odos[k + 1])

    for k in tqdm(range(N)):
        if mk < mK - 1 and timeLsr[mk] <= timeOdo[k + 1]:
            # Force P to symmetric: there are issues with long runs (>10000 steps)
            # seem like the prediction might be introducing some minor asymetries,
            # so best to force P symetric before update (where chol etc. is used).
            # TODO: remove this for short debug runs in order to see if there are small errors
            P = (P + P.T) / 2
            dt = timeLsr[mk] - t
            if dt < 0:  # avoid assertions as they can be optimized avay?
                raise ValueError("negative time increment")

            # ? reset time to this laser time for next post predict
            t = timeLsr[mk]
            odo = odometry(speed[k + 1], steering[k + 1], dt, car)
            eta, P =  slam.predict(eta, P, odo)

            z = detectTrees(LASER[mk])
            eta, P, NIS[mk], a[mk] =  slam.update(eta, P, z)

            num_asso = np.count_nonzero(a[mk] > -1)
            num_assos[mk] = num_asso

            if num_asso > 0:
                NISnorm[mk] = NIS[mk] / num_asso
                CInorm[mk] = np.array(chi2.interval(confidence_prob, 2 * num_asso)) / num_asso
            else:
                NISnorm[mk] = 1
                CInorm[mk].fill(1)

            xupd[mk] = eta[:3]

            if doPlot:
                sh_lmk.set_offsets(eta[3:].reshape(-1, 2))
                if len(z) > 0:
                    zinmap = (
                        rotmat2d(eta[2])
                        @ (
                            z[:, 0] *
                            np.array([np.cos(z[:, 1]), np.sin(z[:, 1])])
                            + slam.sensor_offset[:, None]
                        )
                        + eta[0:2, None]
                    )
                    sh_Z.set_offsets(zinmap.T)
                lh_pose.set_data(*xupd[mk_first:mk, :2].T)

                ax.set(
                    xlim=[-200, 200],
                    ylim=[-200, 200],
                    title=f"step {k}, laser scan {mk}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}",
                )
                plt.draw()
                plt.pause(0.00001)

            mk += 1

        # What if the previous state is closer in time to the next gps measurement?
        # The odometry is so fast updated that this is porbably not an issue 
        if t >= timeGps[gps_ind]:
            pos_err[gps_ind] = np.linalg.norm(eta[:2] - gps[gps_ind])
            gps_nis[gps_ind] = get_gps_nis(eta[:3], P[:3,:3], gps[gps_ind], R_gps, sensorOffset)
            gps_ind += 1

        if k < K - 1:
            dt = timeOdo[k + 1] - t
            t = timeOdo[k + 1]
            odo = odometry(speed[k + 1], steering[k + 1], dt, car)
            eta, P = slam.predict(eta, P, odo)
            

    # %% Consistency
    
    # GPS NIS
    gps_CI = np.tile(chi2.interval(confidence_prob, 2), (gps_ind,1))
    gps_insideCI = (gps_CI[:,0] <= gps_nis[:gps_ind]) * \
        (gps_nis[:gps_ind] <= gps_CI[:,1])
    gps_ANIS = gps_nis[:gps_ind].mean()
    gps_ANIS_CI = np.array(chi2.interval(confidence_prob, 2 * gps_ind)) / gps_ind
    
    
    fig2, ax2 = plt.subplots(nrows=2, ncols=1, num=2, clear=True, figsize=(7, 7))
    ax2[0].plot(gps_CI[:,0], "--")
    ax2[0].plot(gps_CI[:,1], "--")
    ax2[0].plot(gps_nis[:gps_ind], lw=0.5)
    ax2[0].set_title(f'GPS NIS\n {round(gps_insideCI.mean()*100,1):.2f}% inside CI,    ANIS: {round(gps_ANIS,3)}, CI: {gps_ANIS_CI.round(3)}')
    
    pos_err = pos_err[:gps_ind]
    ax2[1].plot(pos_err, label="error")
    ax2[1].set_title(f"Position error,   RMSE: {round(np.sqrt((pos_err**2).mean()),3)} m")
    ax2[1].set_ylabel(f"[m]")
    ax2[1].set_xlabel("GPS measurements")
    
    fig2.canvas.manager.set_window_title("GPS Comparison")
    fig2.savefig(plot_folder.joinpath("GPS Comparison.pdf"))
    
    # NIS
    insideCI = (CInorm[:mk, 0] <= NISnorm[:mk]) * \
        (NISnorm[:mk] <= CInorm[:mk, 1])
    dof = num_assos.sum()
    ANIS = NIS.sum() / dof
    ANIS_CI = np.array(chi2.interval(alpha, 2 * dof)) / dof


    fig3, ax3 = plt.subplots(num=3, clear=True, figsize=(7, 5))
    ax3.plot(CInorm[:mk, 0], "--")
    ax3.plot(CInorm[:mk, 1], "--")
    ax3.plot(NISnorm[:mk], lw=0.5)

    ax3.set_title(f'NIS\n {round(insideCI.mean()*100,1):.2f}% inside CI,    ANIS: {round(ANIS,3)}, CI: {ANIS_CI.round(3)}')
    fig3.canvas.manager.set_window_title("NIS")
    fig3.savefig(plot_folder.joinpath("NIS.pdf"))

    # ## GPS vs estimated track
    # fig4, ax4 = plt.subplots(num=4, clear=True, figsize=(7, 5))
    # ax4.scatter(
    #     Lo_m[timeGps < timeOdo[N - 1]],
    #     La_m[timeGps < timeOdo[N - 1]],
    #     c="r",
    #     marker=".",
    #     label="GPS",
    # )
    # ax4.plot(*xupd[:mk,:2].T, c="g", label="estimate")
    # ax4.plot(*ellipse(xupd[mk-1, :2], P[:2, :2], 5, 200).T, c="g")
    # ax4.grid()
    # ax4.set_title("GPS vs estimated track")
    # ax4.legend()
    # fig4.canvas.manager.set_window_title("GPS vs estimate")
    # fig4.savefig(plot_folder.joinpath("GPS vs estimate.pdf"))

    ## GPS vs odometry
    if do_raw_prediction:
        fig5, ax5 = plt.subplots(num=5, clear=True, figsize=(7, 5))
        ax5.scatter(
            Lo_m[timeGps < timeOdo[N - 1]],
            La_m[timeGps < timeOdo[N - 1]],
            c="r",
            marker=".",
            label="GPS",
        )
        ax5.plot(*odox[:N, :2].T, label="odom")
        ax5.grid()
        ax5.set_title("GPS vs odometry integration")
        ax5.legend()
        fig5.canvas.manager.set_window_title("GPS vs odometry")
        fig5.savefig(plot_folder.joinpath("GPS vs odometry.pdf"))

    # Estimated track and landmarks
    fig6, ax6 = plt.subplots(num=6, clear=True, figsize=(7, 5))
    ax6.scatter(*eta[3:].reshape(-1, 2).T, color="r", marker="x", label="landmarks")
    ax6.scatter(
        Lo_m[timeGps < timeOdo[N - 1]],
        La_m[timeGps < timeOdo[N - 1]],
        c="g",
        marker=".",
        label="GPS",
    )
    ax6.plot(*xupd[mk_first:mk, :2].T, c="b", label="estimate")
    ax6.plot(*ellipse(xupd[mk-1, :2], P[:2, :2], 5, 200).T, c="b")
    ax6.legend()
    ax6.set(
        title=f"Steps {k}, laser scans {mk-1}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}"
    )
    fig6.canvas.manager.set_window_title("Tracking results")
    fig6.savefig(plot_folder.joinpath("Tracking results.pdf"))
    
    if showPlots:
        plt.show()
        

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

if __name__ == "__main__":
    main()
