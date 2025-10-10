import matplotlib.pyplot as plt
"""This function shows Mullens effect for the P4"""
from pvc_tensile_experiment.Functions import *
def StressRelaxationRegionSelector(expTime, stress):
    # preallocate the maximum stress index
    maxIndex = np.zeros([0], dtype = int)

    # find the maximum stress for each region
    for i in range(0, 5):
        regionIndex = np.where(np.logical_and(expTime > i*3800, expTime < (i+1)*3820))[0]

        # find the index of the maximum value and concate to the maximum stress index
        maxIndex = np.hstack([maxIndex, regionIndex[np.where(stress[regionIndex] == np.max(stress[regionIndex]))[0]]])

    # make the start index 5 before the maximum stress
    startIndices =  np.array(maxIndex) - 5
    
    # get the end indices based on the points before the start of the next region
    endIndices = np.array([startIndices[1] - 60, startIndices[2] - 70, startIndices[3] - 60, startIndices[4] - 60, len(stress) - 60])
    regions = np.array([startIndices, endIndices])

    return regions

def ViscoelasticDataProcessor(folderName, name):
    # import the excel file. dont use columns beyond 4 since they're empty  due to 
    # needing a place to put extra comments
    df = pd.read_excel(f"Data/Viscoelastic Data/{folderName}/{name}", header = None, usecols = [0, 1, 2, 3, 4])

    # get sample geometric data. length is in mm and area is in mm^2
    sampleLength = df.loc[5][1]*1e-3 # sample length - converted from mm to m
    sampleArea = df.loc[6][1]*1e-6 # sample area - converted from mm^2 to m^2

    # preallocate the measurement names
    data = df.loc[df.index[32::]].to_numpy(dtype = float)
    
    # dataColumns = ['Time (s)', 'Temp. (Cel)', 'Displacement (m)', 'Load (N)', 'Displacement Rate (m/s)']
    expTime = data[:, 0]*60      # time -  converted from min to sec
    strain = data[:, 2]*1e-6/sampleLength    # displacement - converted from um to m
    stress = data[:, 3]*1e-6/sampleArea     # force - converted from uN to N
    strainRate = data[:, 4]*1e-6/60/sampleLength    # displacement rate - converted from um/min to m/s then to strain/s
    return expTime, strain, strainRate, stress


def RelaxationDataViewer(plastiRatio):

    # define the folder name and pull the files that end with xlsx
    folderName = 'Stress Relaxation Data/Mullens'
    fileNames = [i for i in os.listdir(f'Data/Viscoelastic Data/{folderName}') if i.endswith('.xlsx') and i.find(plastiRatio) != -1]

    # plot parameters
    markerSize = 4
    titleSize = 15
    axisSize = 12
    legendSize = 11

    for i in fileNames:
        # read and process the data file for strain and stress 
        expTime, strain, _, stress = ViscoelasticDataProcessor(folderName, i)
        

        # the starting point is 4 before the maximum stress value
        regions = StressRelaxationRegionSelector(expTime, stress)
        for j in [0,1,2,4]: 
            if j == 3:
                indexRange = range(regions[0, j] + 55, regions[1, j])
            # extract the increasing strain regions. since the start is 5 indices from the maximum stress,
            else:
                indexRange = range(regions[0, j], regions[1, j])

            stressOffset = stress[indexRange][0]
            strainOffset = strain[indexRange][0]

            # the starting point is 4 before the maximum stress value
            regions = StressRelaxationRegionSelector(expTime, stress)

            # reset the index range so the first data point is the maximum stress value.
            if j == 3:
                indexRange = range(regions[0, j] + 65, regions[1, j])
            # extract the increasing strain regions. since the start is 5 indices from the maximum stress,
                strainOffset = strainOffset - 0.007
            else:
                indexRange = range(regions[0, j] + 10, regions[1, j])

            # define the variables 
            expTimeFit = expTime[indexRange] - expTime[indexRange[0]]
            strainFit = strain[indexRange] - strainOffset
            stressFit = stress[indexRange] - stressOffset
            normStressFit = stressFit/strainFit[0]/1.8

            plt.yscale('log' )
            plt.xscale('log')
            plt.scatter(expTimeFit, normStressFit, s = markerSize, label = f'\u03B5: {np.round(strainFit[0]*100, 2)}%')
            plt.xlabel('Time (s)', fontsize = axisSize)
            plt.ylabel('E$_{R}(t)$', fontsize = axisSize) #\u03C3
            plt.title(f'{plastiRatio} PVC Gel Stress Relaxation Modulus', fontsize = titleSize)
            plt.legend(fontsize = legendSize, loc = 'right', bbox_to_anchor = (1.45, 0.5))

        
for plastiRatio in ['P2','P4','P6', 'P8']:
    RelaxationDataViewer(plastiRatio)
    plt.show()

# # --- Helper functions ---
# def draw_spring(ax, x, y, length=1.0, height=0.2, turns=5, orientation='horizontal', label=None):
#     import numpy as np
#     if orientation == 'horizontal':
#         t = np.linspace(0, 1, 100)
#         x_vals = x + t * length
#         y_vals = y + height * np.sin(turns * 2 * np.pi * t)
#         ax.plot(x_vals, y_vals, color='black', lw=2)
#         if label:
#             ax.text(x + length/2, y + 0.3, label, ha='center', va='bottom', fontsize=12)
#     else:
#         t = np.linspace(0, 1, 100)
#         x_vals = x + height * np.sin(turns * 2 * np.pi * t)
#         y_vals = y + t * length
#         ax.plot(x_vals, y_vals, color='black', lw=2)
#         if label:
#             ax.text(x + 0.3, y + length/2, label, ha='left', va='center', fontsize=12)

# def draw_dashpot(ax, x, y, length=1.0, orientation='horizontal', label=None):
#     if orientation == 'horizontal':
#         ax.plot([x, x+length*0.3], [y, y], color='black', lw=2)
#         ax.plot([x+length*0.7, x+length], [y, y], color='black', lw=2)
#         ax.add_patch(plt.Rectangle((x+length*0.3, y-0.1), length*0.4, 0.2,
#                                    fill=False, edgecolor='black', lw=2))
#         if label:
#             ax.text(x + length/2, y + 0.3, label, ha='center', va='bottom', fontsize=12)
#     else:
#         ax.plot([x, x], [y, y+length*0.3], color='black', lw=2)
#         ax.plot([x, x], [y+length*0.7, y+length], color='black', lw=2)
#         ax.add_patch(plt.Rectangle((x-0.1, y+length*0.3), 0.2, length*0.4,
#                                    fill=False, edgecolor='black', lw=2))
#         if label:
#             ax.text(x + 0.3, y + length/2, label, ha='left', va='center', fontsize=12)

# def make_figure(title):
#     fig, ax = plt.subplots(figsize=(6,2))
#     ax.set_title(title, fontsize=14, pad=20)
#     ax.axis('off')
#     return fig, ax


# # --- Standard Linear Solid (SLS) ---
# fig, ax = make_figure("Standard Linear Solid")
# # Parallel top: spring (E1)
# draw_spring(ax, 0, 0.5, length=1, label="E∞")
# ax.plot([-0.5, 0], [0.5, 1], color='white', lw=2)
# ax.plot([-0.5, 0], [0.5, 0.5], color='black', lw=2)
# ax.plot([1, 2.5], [0.5, 0.5], color='black', lw=2)
# # Parallel bottom: Maxwell (spring + dashpot)
# draw_spring(ax, 0, -0.3, length=1, label="E₁")
# draw_dashpot(ax, 1, -0.3, length=1, label="η₁")
# ax.plot([-0.5, 0], [-0.3, -0.3], color='black', lw=2)
# ax.plot([2, 2.5], [-0.3, -0.3], color='black', lw=2)
# # # Left and right verticals
# ax.plot([-0.5, -0.5], [-0.3, 0.5], color='black', lw=2)
# ax.plot([2.5, 2.5], [-0.3, 0.5], color='black', lw=2)
# ax.set_title('Standard Linear Solid (SLS) Model')
# plt.show()

# # --- 2-term Prony Series ---
# fig, ax = make_figure("2-term Prony Series Model")
# # Equilibrium spring
# # Parallel top: spring (E1)
# draw_spring(ax, 0, 0.5, length=1, label="E∞")
# ax.plot([-0.5, 0], [0.5, 1], color='white', lw=2)
# ax.plot([-0.5, 0], [0.5, 0.5], color='black', lw=2)
# ax.plot([1, 2.5], [0.5, 0.5], color='black', lw=2)
# # Parallel bottom: Maxwell (spring + dashpot)
# draw_spring(ax, 0, -0.4, length=1, label="E₁")
# draw_dashpot(ax, 1, -0.4, length=1, label="η₁")
# ax.plot([-0.5, 0], [-0.4, -0.4], color='black', lw=2)
# ax.plot([2, 2.5], [-0.4, -0.4], color='black', lw=2)
# # Maxwell arm 2
# draw_spring(ax, 0, -1.3, length=1, label="E₂")
# draw_dashpot(ax, 1, -1.3, length=1, label="η₂")
# ax.plot([-0.5, 0], [-1.3, -1.3], color='black', lw=2)
# ax.plot([2, 2.5], [-1.3, -1.3], color='black', lw=2)
# # Left and right verticals
# ax.plot([-0.5, -0.5], [-1.3, 0.5], color='black', lw=2)
# ax.plot([2.5, 2.5], [-1.3, 0.5], color='black', lw=2)
# ax.set_title('2-term Prony Series')
# plt.show()

# # --- 3-term Prony Series ---
# fig, ax = make_figure("3-term Prony Series")
# # Equilibrium spring
# # Parallel top: spring (E1)
# draw_spring(ax, 0, 0.5, length=1, label="E∞")
# ax.plot([-0.5, 0], [0.5, 1.1], color='white', lw=2)
# ax.plot([-0.5, 0], [0.5, 0.5], color='black', lw=2)
# ax.plot([1, 2.5], [0.5, 0.5], color='black', lw=2)
# # Parallel bottom: Maxwell (spring + dashpot)
# draw_spring(ax, 0, -0.5, length=1, label="E₁")
# draw_dashpot(ax, 1, -0.5, length=1, label="η₁")
# ax.plot([-0.5, 0], [-0.5, -0.5], color='black', lw=2)
# ax.plot([2, 2.5], [-0.5, -0.5], color='black', lw=2)
# # Maxwell arm 2
# draw_spring(ax, 0, -1.5, length=1, label="E₂")
# draw_dashpot(ax, 1, -1.5, length=1, label="η₂")
# ax.plot([-0.5, 0], [-1.5, -1.5], color='black', lw=2)
# ax.plot([2, 2.5], [-1.5, -1.5], color='black', lw=2)
# # Maxwell arm 3
# draw_spring(ax, 0, -2.5, length=1, label="E₃")
# draw_dashpot(ax, 1, -2.5, length=1, label="η₃")
# ax.plot([-0.5, 0], [-2.5, -2.5], color='black', lw=2)
# ax.plot([2, 2.5], [-2.5, -2.5], color='black', lw=2)
# # Left and right verticals
# ax.plot([-0.5, -0.5], [-2.5, 0.5], color='black', lw=2)
# ax.plot([2.5, 2.5], [-2.5, 0.5], color='black', lw=2)
# ax.set_title('3-term Prony Series')

# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Time array
t = np.linspace(0, 2*np.pi, 50)

# Parameters
strain_amp = 1.0
omega = 1.0
delta = np.pi / 4  # phase lag (45 degrees)

# Define signals
strain = strain_amp * np.sin(omega * t)
stress_elastic = strain_amp * np.sin(omega * t)
stress_viscous = strain_amp * np.sin(omega * t + np.pi/2)
stress_viscoelastic = strain_amp * np.sin(omega * t + delta)

# Save a high-resolution (300 dpi) version of the viscoelastic stress response figure as PNG

fig, ax = plt.subplots(figsize=(10, 6))

# Plot curves
ax.plot(t, strain, linewidth = 1.5, c = 'k', label='Strain, $\\varepsilon(t)$')
ax.scatter(t, stress_elastic, s = 25, label='Elastic Stress, $\\sigma_E(t)$')
ax.scatter(t, stress_viscous, s = 25, label='Viscous Stress, $\\sigma_V(t)$')
ax.scatter(t, stress_viscoelastic, s = 25, label='Viscoelastic Stress, $\\sigma_{VE}(t)$')

# Axis labels and formatting
ax.set_xlabel("Time (rad)", fontsize=12)
ax.set_ylabel("Normalized Amplitude", fontsize=12)
ax.set_title("Elastic and Viscous Stress Response under Oscillatory Strain", fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, ls='--', alpha=0.6)

plt.tight_layout()
plt.show()
