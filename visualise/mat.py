import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
import numpy as np
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from mat_utils import brighten_color, get_latest_robot_position_file
import sys

# --- Initial parameters ---
directory = "../controllers/supervisorGA_starter/data/"
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = get_latest_robot_position_file(directory)
csv_file = f"{directory}{filename}"
gen_filename = filename.replace("robot_positions_", "generation_data_")
gen_csv_file = f"{directory}{gen_filename}"
df_info = pd.read_csv(csv_file, usecols=["generation", "population"])
df_new_global = None  # for hover annotation
df_gen_info = pd.read_csv(gen_csv_file)
avg_fitnesses = df_gen_info.groupby("population")["fitness"].mean()

gen_min, gen_max = df_info["generation"].min(), df_info["generation"].max()
print(f"Generations: {gen_min} to {gen_max}")
populations_all = sorted(df_info["population"].unique())
initial_gen = gen_min
initial_step = 30
show_orientation = False
is_selecting_bests = False
selected_pops = populations_all  # default = all
select_top_count = 5
MARKER_SIZE = 30  # increased for single-point-per-pop animation
MARKER_OPACITY = 0.2
SELECT_REAL_TOP = True

# --- Load first generation ---
df_all = pd.read_csv(csv_file)
norm = colors.Normalize(vmin=df_all["fitness"].min(), vmax=df_all["fitness"].max())
df = df_all[df_all["generation"] == initial_gen]


def load_generation_mem(generation, pops, step):
    """Return rows for generation, optionally filtered by populations, downsampled by step."""
    dfg = df_all[df_all["generation"] == generation]
    if pops:
        dfg = dfg[dfg["population"].isin(pops)]
    return dfg.iloc[::step]

def load_generation_info_mem(generation):
    """Return generation info row for generation."""
    dfg = df_gen_info[df_gen_info["generation"] == generation]
    return dfg


# --- Plot setup ---
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.40, right=0.7)  # space for widgets

sc = ax.scatter(
    df["x"],
    df["y"],
    c=df["fitness"],
    cmap="viridis",
    s=8,
    alpha=0.5,
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"Generation {initial_gen}")
ax.set_xbound(-0.9, 0.9)
ax.set_ybound(-0.9, 0.9)

# Orientation arrows (for static / animation frames)
arrows = []

# --- Slider for generation ---
ax_gen = plt.axes([0.1, 0.2, 0.65, 0.03])
slider_gen = Slider(
    ax_gen, "Generation", gen_min, gen_max, valinit=initial_gen, valstep=1
)

# --- Slider for downsampling ---
ax_step = plt.axes([0.1, 0.15, 0.65, 0.03])
slider_step = Slider(ax_step, "Downsample Step", 1, 50, valinit=initial_step, valstep=1)


# --- Slider for population (single numeric picker) ---
ax_pop = plt.axes([0.1, 0.10, 0.65, 0.03])
slider_pop = Slider(
    ax_pop,
    "Population ID",
    populations_all[0],
    populations_all[-1],
    valinit=populations_all[0],
    valstep=1,
)

# --- Checkbox for orientation ---
ax_select_bests_check = plt.axes([0.8, 0.90, 0.12, 0.05])
orientation_check = CheckButtons(
    ax_select_bests_check, ["Orientation"], [show_orientation]
)

# --- Multi-selector for populations (checkbox list) ---
ax_pop_list = plt.axes([0.80, 0.35, 0.15, 0.50])
pop_labels = [str(p) + "  " for p in populations_all]
selected_pop_states = [False] * len(pop_labels)
pop_check = CheckButtons(ax_pop_list, pop_labels, selected_pop_states)

# --- Button for selecting top individuals ---
ax_select_bests_check = plt.axes([0.8, 0.25, 0.15, 0.05])
select_bests_check = CheckButtons(
    ax_select_bests_check, [f"Select Top {select_top_count}"], [is_selecting_bests]
)

# --- Button for selecting all individuals ---
ax_select_all_button = plt.axes([0.8, 0.18, 0.15, 0.05])
select_all_button = Button(ax_select_all_button, "Select All")

# --- Button for clearing selections ---
ax_clear_button = plt.axes([0.8, 0.12, 0.15, 0.05])
clear_button = Button(ax_clear_button, "Clear Selections")

# --- Animation controls (NEW) ---
ax_animate_check = plt.axes([0.8, 0.06, 0.15, 0.05])
animate_check = CheckButtons(ax_animate_check, ["Animate"], [False])

ax_frame = plt.axes([0.1, 0.01, 0.65, 0.03])
slider_anim_frame = Slider(ax_frame, "Animation Frame", 0, 0, valinit=0, valstep=1)

# ax_play_button = plt.axes([0.8, 0.00, 0.07, 0.04])
# play_button = Button(ax_play_button, "Play")

ax_step_button = plt.axes([0.88, 0.00, 0.07, 0.04])
step_button = Button(ax_step_button, "Step")

# --- Animation internal state ---
anim = None
anim_running = False
anim_frame = 0
anim_data = {}  # map population -> list of dicts with x,y,fitness,ox,oy
anim_max_frames = 0
anim_generation_locked = (
    initial_gen  # which generation animates (kept in sync with slider_gen at start)
)

# --- Hover annotation ---
annot = ax.annotate(
    "",
    xy=(0, 0),
    xytext=(20, -80),
    textcoords="offset points",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
    arrowprops=dict(arrowstyle="->"),
)
annot.set_visible(False)
hide_annot_button = Button(plt.axes([0.8, 0.00, 0.15, 0.04]), "Hide Annotation")

def build_animation_data(generation, selected_pop_list, per_pop_downsample=1):
    """
    Build anim_data and anim_max_frames for the current generation and selection.
    We collect the sequence of positions for each population (in CSV order).
    per_pop_downsample lets us skip frames inside the population sequences if desired.
    """
    global anim_data, anim_max_frames
    anim_data = {}
    anim_max_frames = 0
    # Get rows for this generation but don't downsample here (we want full per-pop sequences)
    df_gen = df_all[df_all["generation"] == generation]
    if selected_pop_list:
        df_gen = df_gen[df_gen["population"].isin(selected_pop_list)]
    # keep original order within df_gen — assume sequence corresponds to robot positions
    for pop, grp in df_gen.groupby("population"):
        seq = grp.iloc[
            ::per_pop_downsample
        ]  # downsample inside population if requested
        frames = []
        for _, row in seq.iterrows():
            frames.append(
                {
                    "x": row["x"],
                    "y": row["y"],
                    "fitness": row.get("fitness", np.nan),
                    "ox": row.get("ox", 0.0),
                    "oy": row.get("oy", 0.0),
                }
            )
        anim_data[pop] = frames
        if len(frames) > anim_max_frames:
            anim_max_frames = len(frames)
        # Update the animation frame slider limits dynamically
        slider_anim_frame.valmin = 0
        slider_anim_frame.valmax = anim_max_frames - 1
        slider_anim_frame.ax.set_xlim(
            slider_anim_frame.valmin, slider_anim_frame.valmax
        )
        slider_anim_frame.set_val(0)  # reset to start
    if anim_max_frames == 0:
        anim_max_frames = 1

def update(val):
    """
    Original update used for static view. Also rebuilds animation data when generation
    or selection changes so animation uses the current data.
    """
    global anim_generation_locked
    gen = int(slider_gen.val)
    step = int(slider_step.val)

    # Build selected populations list from checkboxes + slider_pop
    selected_pops_local = [
        int(lbl.strip())
        for lbl, state in zip(pop_labels, pop_check.get_status())
        if state
    ]
    # include slider_pop single picker as well
    try:
        selected_pops_local.append(int(slider_pop.val))
    except Exception:
        pass
    # unique
    selected_pops_local = sorted(set(selected_pops_local))

    # If "select top" is checked, override selection to top populations
    if select_bests_check.get_status()[0]:
        df_selecting_info = load_generation_info_mem(gen)

        avg_fitnesses = df_selecting_info.groupby("population")["fitness"].mean()
        top = avg_fitnesses.nlargest(select_top_count).index.tolist()
        # Update pop_check buttons (sync UI)
        new_states = [p in top for p in populations_all]
        for i, state in enumerate(new_states):
            # toggle to desired state if necessary
            if state != pop_check.get_status()[i]:
                pop_check.set_active(i)
        selected_pops_local = sorted(set(top))

    # For static view, downsample rows using step
    df_new = load_generation_mem(gen, selected_pops_local, step)
    df_new_info = load_generation_info_mem(gen)

    # Update scatter for static view
    if not df_new.empty:
        sc.set_offsets(df_new[["x", "y"]].values)
        sc.set_array(df_new["fitness"].values)
        sc.set_sizes(
            np.full(len(df_new), MARKER_SIZE / 3)
        )  # small markers for static view
        sc.set_alpha(MARKER_OPACITY / 2)
    else:
        sc.set_offsets(np.empty((0, 2)))
        sc.set_array(np.array([]))

    # ax.set_title(
    #     f"Generation {gen} — {len(selected_pops_local)} populations — {len(df_new)} points"
    # )
    
    ax.set_title(
        f"Generation {gen} — {len(selected_pops_local)} populations — {len(df_new)} points",
        pad=20,
    )

    # Remove old arrows
    global arrows
    for arr in arrows:
        try:
            arr.remove()
        except Exception:
            pass
    arrows = []

    # Add new arrows for static view if orientation checked (use df_new)
    if orientation_check.get_status()[0] and not df_new.empty:
        for _, row in df_new.iterrows():
            arr = ax.arrow(
                row["x"],
                row["y"],
                row.get("ox", 0.0) * 0.05,
                row.get("oy", 0.0) * 0.05,
                head_width=0.01,
                head_length=0.02,
                fc="gray",
                ec="gray",
                alpha=0.6,
            )
            arrows.append(arr)

    # --- Compute average fitness per population for labels (based on df_new) ---
    avg_fitnesses = {}
    if not df_new.empty:
        avg_fitnesses = df_new_info.groupby("population")["fitness"].mean().to_dict()

    for i in range(len(pop_labels)):
        pop_check.labels[i].set_fontsize(8)
        pop_check.labels[i].set_fontweight("bold")

    for i, label in enumerate(pop_check.labels):
        p = populations_all[i]
        fitness = avg_fitnesses.get(p, 0)
        label.set_text(f"{p} ({fitness:.2f})")
        label.set_fontweight("bold")
        color = plt.cm.viridis(norm(fitness))
        label.set_backgroundcolor(brighten_color(color, factor=0.7))

    # Rebuild animation data (so animation frames reflect the current selection & generation)
    # Keep generation locked to current slider value for animation
    anim_generation_locked = gen
    # For per-population sequences we do not downsample by the global step; pass 1.
    if animate_check.get_status()[0]:
        build_animation_data(
            anim_generation_locked, selected_pops_local, per_pop_downsample=1
        )

    fig.canvas.draw_idle()

    global df_new_global
    df_new_global = df_new


# --- Button callback to select all ---
def select_all(event):
    for i in range(len(pop_labels)):
        if not pop_check.get_status()[i]:
            pop_check.set_active(i)
    update(None)


# --- Button callback to clear selections ---
def clear_selections(event):
    select_bests_check.set_active(0, False)  # uncheck "select top"
    for i in range(len(pop_labels)):
        if pop_check.get_status()[i]:
            pop_check.set_active(i)
    update(None)


# --- Animation functions (NEW) ---
def render_animation_frame(frame_idx):
    """
    Render a single animation frame: show for each selected population the position at index frame_idx
    (if that population has that many frames).
    """
    pts = []
    fitnesses = []
    arrows_local = []
    for pop, frames in anim_data.items():
        if frame_idx < len(frames) and len(frames) > 0:
            fr = frames[frame_idx]
            pts.append((fr["x"], fr["y"]))
            fitnesses.append(fr.get("fitness", np.nan))
    if pts:
        pts_arr = np.array(pts)
        sc.set_offsets(pts_arr)
        sc.set_array(np.array(fitnesses))
        sc.set_sizes(np.full(len(pts), MARKER_SIZE))
        sc.set_alpha(MARKER_OPACITY)
    else:
        sc.set_offsets(np.empty((0, 2)))
        sc.set_array(np.array([]))

    # Remove previous arrows
    global arrows
    for arr in arrows:
        try:
            arr.remove()
        except Exception:
            pass
    arrows = []

    if orientation_check.get_status()[0]:
        # draw arrows for this frame
        idx = 0
        for pop, frames in anim_data.items():
            if frame_idx < len(frames) and len(frames) > 0:
                fr = frames[frame_idx]
                arr = ax.arrow(
                    fr["x"],
                    fr["y"],
                    fr.get("ox", 0.0) * 0.05,
                    fr.get("oy", 0.0) * 0.05,
                    head_width=0.01,
                    head_length=0.02,
                    fc="gray",
                    ec="gray",
                    alpha=0.6,
                )
                arrows.append(arr)
                idx += 1

    ax.set_title(
        f"Generation {anim_generation_locked} — anim frame {frame_idx + 1}/{anim_max_frames}"
    )
    # Always show tooltip if only 1 point on the animation
    if len(pts) == 1:
        update_annot(0, df_all, pts[0][0], pts[0][1])
        annot.set_visible(True)
    fig.canvas.draw_idle()


def animate_frame(i):
    global anim_frame
    anim_frame = i % max(1, anim_max_frames)
    render_animation_frame(anim_frame)
    slider_anim_frame.set_val(anim_frame)
    return (sc,)

def toggle_animation(label):
    # when checkbox toggled
    if animate_check.get_status()[0]:
        # ensure anim_data is present (rebuild if necessary)
        build_animation_data(
            slider_gen.val,
            [
                int(lbl.strip())
                for lbl, state in zip(pop_labels, pop_check.get_status())
                if state
            ]
            + [int(slider_pop.val)],
            per_pop_downsample=1,
        )
    else:
        # stop_animation()
        # restore static rendering for current generation & selection
        update(None)

def step_once(event):
    # advance one animation frame (works even if not animating)
    global anim_frame
    # ensure anim_data exists
    if anim_data is None or len(anim_data) == 0:
        # nothing to step
        return
    anim_frame = (anim_frame + 1) % max(1, anim_max_frames)
    render_animation_frame(anim_frame)


def on_anim_slider_changed(val):
    global anim_frame
    anim_frame = int(val)
    render_animation_frame(anim_frame)

def update_annot(idx, df_source, x, y):
    """Update annotation text and position."""
    # get real row from df_source (df_new in static mode, or anim frame data)
    row = df_source.iloc[idx]

    # One index can give multiple rows if there are duplicates; take the first
    row_count = len(df_source[(df_source["x"] == x) & (df_source["y"] == y)])

    text = (
        f"Population: {row['population']}\n"
        f"Fitness: {row['fitness']:.3f}\n"
        f"Forward: {row['forward_fitness']:.3f}\n"
        f"Line: {row['line_fitness']:.3f}\n"
        f"Collision: {row['collision_fitness']:.3f}\n"
        f"Spinning: {row['spinning_fitness']:.3f}\n"
        f"X: {row['x']:.3f}\n"
        f"Y: {row['y']:.3f}\n"
        f"Points at this position: {row_count}"
    )

    annot.xy = (x, y)
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor("white")
    annot.get_bbox_patch().set_alpha(0.9)

def on_hover(event):
    # if event.inaxes != ax:
    #     annot.set_visible(False)
    #     fig.canvas.draw_idle()
    #     return

    cont, ind = sc.contains(event)
    if cont:
        # show values from the currently displayed df
        idx = ind["ind"][0]  # first point
        x, y = sc.get_offsets()[idx]
        update_annot(idx, df_new_global, x, y)
        annot.set_visible(True)
        fig.canvas.draw_idle()
    # else:
    #     if annot.get_visible():
    #         annot.set_visible(False)
    #         fig.canvas.draw_idle()

def hide_annot(event):
    annot.set_visible(False)
    fig.canvas.draw_idle()

# Connect controls
slider_gen.on_changed(update)
slider_step.on_changed(update)
slider_pop.on_changed(update)
orientation_check.on_clicked(lambda label: update(None))
pop_check.on_clicked(lambda label: update(None))
select_bests_check.on_clicked(lambda label: update(None))
select_all_button.on_clicked(select_all)
clear_button.on_clicked(clear_selections)

animate_check.on_clicked(toggle_animation)
step_button.on_clicked(step_once)
slider_anim_frame.on_changed(on_anim_slider_changed)
hide_annot_button.on_clicked(hide_annot)

fig.canvas.mpl_connect("motion_notify_event", on_hover)


# initial build
update(None)

plt.colorbar(sc, ax=ax, label="Fitness")
plt.show()
