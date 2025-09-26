import matplotlib.pyplot as plt

# --- Helper functions ---
def draw_spring(ax, x, y, length=1.0, height=0.2, turns=5, orientation='horizontal', label=None):
    import numpy as np
    if orientation == 'horizontal':
        t = np.linspace(0, 1, 100)
        x_vals = x + t * length
        y_vals = y + height * np.sin(turns * 2 * np.pi * t)
        ax.plot(x_vals, y_vals, color='black', lw=2)
        if label:
            ax.text(x + length/2, y + 0.3, label, ha='center', va='bottom', fontsize=12)
    else:
        t = np.linspace(0, 1, 100)
        x_vals = x + height * np.sin(turns * 2 * np.pi * t)
        y_vals = y + t * length
        ax.plot(x_vals, y_vals, color='black', lw=2)
        if label:
            ax.text(x + 0.3, y + length/2, label, ha='left', va='center', fontsize=12)

def draw_dashpot(ax, x, y, length=1.0, orientation='horizontal', label=None):
    if orientation == 'horizontal':
        ax.plot([x, x+length*0.3], [y, y], color='black', lw=2)
        ax.plot([x+length*0.7, x+length], [y, y], color='black', lw=2)
        ax.add_patch(plt.Rectangle((x+length*0.3, y-0.1), length*0.4, 0.2,
                                   fill=False, edgecolor='black', lw=2))
        if label:
            ax.text(x + length/2, y + 0.3, label, ha='center', va='bottom', fontsize=12)
    else:
        ax.plot([x, x], [y, y+length*0.3], color='black', lw=2)
        ax.plot([x, x], [y+length*0.7, y+length], color='black', lw=2)
        ax.add_patch(plt.Rectangle((x-0.1, y+length*0.3), 0.2, length*0.4,
                                   fill=False, edgecolor='black', lw=2))
        if label:
            ax.text(x + 0.3, y + length/2, label, ha='left', va='center', fontsize=12)

def make_figure(title):
    fig, ax = plt.subplots(figsize=(6,2))
    ax.set_title(title, fontsize=14, pad=20)
    ax.axis('off')
    return fig, ax


# --- Standard Linear Solid (SLS) ---
fig, ax = make_figure("Standard Linear Solid")
# Parallel top: spring (E1)
draw_spring(ax, 0, 0.5, length=1, label="E∞")
ax.plot([-0.5, 0], [0.5, 1], color='white', lw=2)
ax.plot([-0.5, 0], [0.5, 0.5], color='black', lw=2)
ax.plot([1, 2.5], [0.5, 0.5], color='black', lw=2)
# Parallel bottom: Maxwell (spring + dashpot)
draw_spring(ax, 0, -0.3, length=1, label="E₁")
draw_dashpot(ax, 1, -0.3, length=1, label="η₁")
ax.plot([-0.5, 0], [-0.3, -0.3], color='black', lw=2)
ax.plot([2, 2.5], [-0.3, -0.3], color='black', lw=2)
# # Left and right verticals
ax.plot([-0.5, -0.5], [-0.3, 0.5], color='black', lw=2)
ax.plot([2.5, 2.5], [-0.3, 0.5], color='black', lw=2)
ax.set_title('Standard Linear Solid (SLS) Model')
plt.show()

# --- 2-term Prony Series ---
fig, ax = make_figure("2-term Prony Series Model")
# Equilibrium spring
# Parallel top: spring (E1)
draw_spring(ax, 0, 0.5, length=1, label="E∞")
ax.plot([-0.5, 0], [0.5, 1], color='white', lw=2)
ax.plot([-0.5, 0], [0.5, 0.5], color='black', lw=2)
ax.plot([1, 2.5], [0.5, 0.5], color='black', lw=2)
# Parallel bottom: Maxwell (spring + dashpot)
draw_spring(ax, 0, -0.4, length=1, label="E₁")
draw_dashpot(ax, 1, -0.4, length=1, label="η₁")
ax.plot([-0.5, 0], [-0.4, -0.4], color='black', lw=2)
ax.plot([2, 2.5], [-0.4, -0.4], color='black', lw=2)
# Maxwell arm 2
draw_spring(ax, 0, -1.3, length=1, label="E₂")
draw_dashpot(ax, 1, -1.3, length=1, label="η₂")
ax.plot([-0.5, 0], [-1.3, -1.3], color='black', lw=2)
ax.plot([2, 2.5], [-1.3, -1.3], color='black', lw=2)
# Left and right verticals
ax.plot([-0.5, -0.5], [-1.3, 0.5], color='black', lw=2)
ax.plot([2.5, 2.5], [-1.3, 0.5], color='black', lw=2)
ax.set_title('2-term Prony Series')
plt.show()

# --- 3-term Prony Series ---
fig, ax = make_figure("3-term Prony Series")
# Equilibrium spring
# Parallel top: spring (E1)
draw_spring(ax, 0, 0.5, length=1, label="E∞")
ax.plot([-0.5, 0], [0.5, 1.1], color='white', lw=2)
ax.plot([-0.5, 0], [0.5, 0.5], color='black', lw=2)
ax.plot([1, 2.5], [0.5, 0.5], color='black', lw=2)
# Parallel bottom: Maxwell (spring + dashpot)
draw_spring(ax, 0, -0.5, length=1, label="E₁")
draw_dashpot(ax, 1, -0.5, length=1, label="η₁")
ax.plot([-0.5, 0], [-0.5, -0.5], color='black', lw=2)
ax.plot([2, 2.5], [-0.5, -0.5], color='black', lw=2)
# Maxwell arm 2
draw_spring(ax, 0, -1.5, length=1, label="E₂")
draw_dashpot(ax, 1, -1.5, length=1, label="η₂")
ax.plot([-0.5, 0], [-1.5, -1.5], color='black', lw=2)
ax.plot([2, 2.5], [-1.5, -1.5], color='black', lw=2)
# Maxwell arm 3
draw_spring(ax, 0, -2.5, length=1, label="E₃")
draw_dashpot(ax, 1, -2.5, length=1, label="η₃")
ax.plot([-0.5, 0], [-2.5, -2.5], color='black', lw=2)
ax.plot([2, 2.5], [-2.5, -2.5], color='black', lw=2)
# Left and right verticals
ax.plot([-0.5, -0.5], [-2.5, 0.5], color='black', lw=2)
ax.plot([2.5, 2.5], [-2.5, 0.5], color='black', lw=2)
ax.set_title('3-term Prony Series')

plt.show()

