import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
import numpy as np
from line_profiler import profile
import colorsys
from matplotlib import colors

### THIS FILE HAS BEEN AI ASSISTED ###

def brighten_color(color, factor=0.3):
    r, g, b, a = color
    h, light, s = colorsys.rgb_to_hls(r, g, b)
    light = min(1, light + factor * (1 - light))  # smoothly increase lightness
    r, g, b = colorsys.hls_to_rgb(h, light, s)
    return (r, g, b, a)

# --- Initial parameters ---
filename = "robot_position_20251109-030429"
csv_file = f"../controllers/supervisorGA_starter/data/{filename}.csv"
df_info = pd.read_csv(csv_file, usecols=["generation", "population"])
gen_min, gen_max = df_info["generation"].min(), df_info["generation"].max()
print(f"Generations: {gen_min} to {gen_max}")
populations_all = sorted(df_info["population"].unique())
initial_gen = gen_min
initial_step = 30
show_orientation = False
is_selecting_bests = False
selected_pops = populations_all  # default = all
select_top_count = 5
MARKER_SIZE = 8
MARKER_OPACITY = 0.5
SELECT_REAL_TOP = True

# --- Load first generation ---
df_all = pd.read_csv(csv_file)
norm = colors.Normalize(vmin=df_all["fitness"].min(), vmax=df_all["fitness"].max())
df = df_all[df_all["generation"] == initial_gen]
def load_generation_mem(generation, pops, step):
    df = df_all[df_all["generation"] == generation]
    if pops:
        df = df[df["population"].isin(pops)]
    return df.iloc[::step]

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.40, right=0.7)  # space for widgets

sc = ax.scatter(
    df["x"],
    df["y"],
    c=df["fitness"],
    cmap="viridis",
    s=MARKER_SIZE,
    alpha=MARKER_OPACITY,  # set alpha <1
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"Generation {initial_gen}")
ax.set_xbound(-0.9, 0.9)
ax.set_ybound(-0.9, 0.9)

# Orientation arrows
arrows = []
if show_orientation:
    for _, row in df.iterrows():
        arr = ax.arrow(
            row["x"],
            row["y"],
            row["ox"] * 0.05,
            row["oy"] * 0.05,
            head_width=0.01,
            head_length=0.02,
            fc="gray",
            ec="gray",
            alpha=0.6,
        )
        arrows.append(arr)

# --- Slider for generation ---
ax_gen = plt.axes([0.1, 0.2, 0.65, 0.03])
slider_gen = Slider(
    ax_gen, "Generation", gen_min, gen_max, valinit=initial_gen, valstep=1
)

# --- Slider for downsampling ---
ax_step = plt.axes([0.1, 0.15, 0.65, 0.03])
slider_step = Slider(ax_step, "Downsample Step", 1, 50, valinit=initial_step, valstep=1)


# --- Slider for population ---
ax_pop = plt.axes([0.1, 0.10, 0.65, 0.03])
slider_pop = Slider(
    ax_pop,
    "Population ID (min-max)",
    populations_all[0],
    populations_all[-1],
    valinit=populations_all[0],
    valstep=1,
)

# --- Checkbox for orientation ---
ax_select_bests_check = plt.axes([0.8, 0.90, 0.12, 0.05])
orientation_check = CheckButtons(ax_select_bests_check, ["Orientation"], [show_orientation])

# --- Multi-selector for populations ---
ax_pop = plt.axes([0.85, 0.35, 0.06, 0.50])
# We'll just simulate checkboxes via CheckButtons for simplicity
pop_labels = [str(p) + "  " for p in populations_all]
selected_pop_states = [False] * len(pop_labels)
pop_check = CheckButtons(ax_pop, pop_labels, selected_pop_states)

population_fitness_labels = []
for p in populations_all:
    population_fitness_labels.append(
        ax.text(
            0.82,
            0.82 - populations_all.index(p) * 0.0237,
            f"{p}",
            transform=fig.transFigure,
            fontsize=8,
        )
    )

# --- Button for selecting top individuals ---
ax_select_bests_check = plt.axes([0.8, 0.25, 0.15, 0.05])
select_bests_check = CheckButtons(ax_select_bests_check, [f"Select Top {select_top_count}"], [is_selecting_bests])


# --- Button for selecting all individuals ---
ax_select_all_button = plt.axes([0.8, 0.18, 0.15, 0.05])
select_all_button = Button(ax_select_all_button, "Select All")

# --- Button for clearing selections ---
ax_clear_button = plt.axes([0.8, 0.12, 0.15, 0.05])
clear_button = Button(ax_clear_button, "Clear Selections")


# --- Update function ---
@profile
def update(val):
    gen = int(slider_gen.val)
    step = int(slider_step.val)

    selected_pops = [
        int(lbl) for lbl, state in zip(pop_labels, pop_check.get_status()) if state
    ]
    selected_pops.append(int(slider_pop.val))

    if select_bests_check.get_status()[0]:
        df_selecting = load_generation_mem(gen, [], 1 if SELECT_REAL_TOP else step)
        avg_fitnesses = df_selecting.groupby("population")["fitness"].mean()
        top = avg_fitnesses.nlargest(select_top_count).index.tolist()
        # Update pop_check buttons
        new_states = [p in top for p in populations_all]
        for i, state in enumerate(new_states):
            pop_check.set_active(i) if state != pop_check.get_status()[i] else None
        selected_pops = top

    df_new = load_generation_mem(gen, selected_pops, step)

    # Update scatter
    if not df_new.empty:
        sc.set_offsets(df_new[["x", "y"]].values)
        sc.set_array(df_new["fitness"].values)
        sc.set_alpha(MARKER_OPACITY)
    else:
        sc.set_offsets(np.empty((0, 2)))
        sc.set_array(np.array([]))

    ax.set_title(
        f"Generation {gen} — {len(selected_pops)} populations — {len(df_new)} points"
    )

    # Remove old arrows
    global arrows
    for arr in arrows:
        arr.remove()
    arrows = []

    # Add new arrows if orientation checked
    if orientation_check.get_status()[0] and not df_new.empty:
        for _, row in df_new.iterrows():
            arr = ax.arrow(
                row["x"],
                row["y"],
                row["ox"] * 0.05,
                row["oy"] * 0.05,
                head_width=0.01,
                head_length=0.02,
                fc="gray",
                ec="gray",
                alpha=0.6,
            )
            arrows.append(arr)

    # --- Compute average fitness per population ---
    if not df_new.empty:
        avg_fitnesses = df_new.groupby("population")["fitness"].mean()

    for i in range(len(pop_labels)):
        pop_check.labels[i].set_fontsize(8)
        pop_check.labels[i].set_fontweight("bold")

    for i, label in enumerate(population_fitness_labels):
        p = populations_all[i]
        fitness = avg_fitnesses.get(p, 0)
        label.set_text(f"{fitness:.2f}")
        label.set_fontweight("bold")
        color = plt.cm.viridis(norm(fitness))
        label.set_backgroundcolor(brighten_color(color, factor=0.7))

    fig.canvas.draw_idle()


# --- Button callback to select all ---
def select_all(event):
    for i in range(len(pop_labels)):
        if not pop_check.get_status()[i]:
            pop_check.set_active(i)
    update(None)


# --- Button callback to clear selections ---
def clear_selections(event):
    for i in range(len(pop_labels)):
        if pop_check.get_status()[i]:
            pop_check.set_active(i)


update(None)

slider_gen.on_changed(update)
slider_step.on_changed(update)
slider_pop.on_changed(update)
orientation_check.on_clicked(lambda label: update(None))
pop_check.on_clicked(lambda label: update(None))
select_bests_check.on_clicked(lambda label: update(None))
select_all_button.on_clicked(select_all)
clear_button.on_clicked(clear_selections)

plt.colorbar(sc, ax=ax, label="Fitness")
plt.show()
