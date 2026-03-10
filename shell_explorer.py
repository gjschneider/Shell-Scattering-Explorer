import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons

def sphere_form_factor(q, r):
    """Normalized form factor for a sphere."""
    if r <= 0: return np.zeros_like(q)
    qr = q * r
    # Avoid division by zero at q=0
    mask = qr == 0
    ff = np.ones_like(qr)
    
    # Optimization: slice the array once rather than 4 times
    valid_qr = qr[~mask]
    ff[~mask] = 3 * (np.sin(valid_qr) - valid_qr * np.cos(valid_qr)) / (valid_qr**3)
    return ff

def get_sld_profile(core_t, in_peg_t, pbo_t, out_peg_t, core_sld, pbo_sld, peg_sld, sol_sld):
    """Generates the radial SLD profile data for a Vesicle visualization."""
    r_core = core_t
    r_in_peg = r_core + in_peg_t
    r_pbo = r_in_peg + pbo_t
    r_out_peg = r_pbo + out_peg_t
    r_end = r_out_peg + 50  # Add some extra space to show the solvent baseline
    
    # Create step-like arrays for the radius and SLD
    # Sequence: Core -> Inner PEG -> PBO -> Outer PEG -> Solvent
    r = [0, r_core, r_core, r_in_peg, r_in_peg, r_pbo, r_pbo, r_out_peg, r_out_peg, r_end]
    sld = [core_sld, core_sld, peg_sld, peg_sld, pbo_sld, pbo_sld, peg_sld, peg_sld, sol_sld, sol_sld]
    return r, sld

def calculate_components(q, solvent_sld, core_t, in_peg_t, pbo_t, out_peg_t, core_sld, pbo_sld, peg_sld,
                         include_core=True, include_in_peg=True, include_pbo=True, include_out_peg=True,
                         use_cross_terms=True):
    """Calculates SANS intensity components for the D2O/PEO/PBO/PEO Vesicle structure."""
    r_core = core_t
    r_in_peg = r_core + in_peg_t
    r_pbo = r_in_peg + pbo_t
    r_out_peg = r_pbo + out_peg_t
    
    # Volumes
    v_core = (4/3) * np.pi * r_core**3
    v_in_peg = (4/3) * np.pi * r_in_peg**3
    v_pbo = (4/3) * np.pi * r_pbo**3
    v_out_peg = (4/3) * np.pi * r_out_peg**3
    
    # Amplitudes of individual physical layers (relative to solvent baseline)
    a_core    = (core_sld - solvent_sld) * (v_core * sphere_form_factor(q, r_core))
    
    a_in_peg  = (peg_sld - solvent_sld) * (v_in_peg * sphere_form_factor(q, r_in_peg) -
                                           v_core * sphere_form_factor(q, r_core))
                                           
    a_pbo     = (pbo_sld - solvent_sld) * (v_pbo * sphere_form_factor(q, r_pbo) -
                                           v_in_peg * sphere_form_factor(q, r_in_peg))
                                           
    a_out_peg = (peg_sld - solvent_sld) * (v_out_peg * sphere_form_factor(q, r_out_peg) -
                                           v_pbo * sphere_form_factor(q, r_pbo))
    
    # Apply toggles to mathematically remove layers from the total calculation
    a_core_eff    = a_core if include_core else np.zeros_like(q)
    a_in_peg_eff  = a_in_peg if include_in_peg else np.zeros_like(q)
    a_pbo_eff     = a_pbo if include_pbo else np.zeros_like(q)
    a_out_peg_eff = a_out_peg if include_out_peg else np.zeros_like(q)
    
    # Total Intensity calculation
    total_coherent = (np.abs(a_core_eff + a_in_peg_eff + a_pbo_eff + a_out_peg_eff)**2) * 1e-8
    total_incoherent = (np.abs(a_core_eff)**2 + np.abs(a_in_peg_eff)**2 + np.abs(a_pbo_eff)**2 + np.abs(a_out_peg_eff)**2) * 1e-8
    
    if use_cross_terms:
        total_intensity = total_coherent
    else:
        total_intensity = total_incoherent
        
    # Net Cross Terms (absolute magnitude of interference, plotted for visualization)
    cross_term_intensity = np.abs(total_coherent - total_incoherent)
    
    # Floor intensities to prevent Log(0) errors when components vanish completely
    total_intensity = np.maximum(total_intensity, 1e-16)
    
    # Individual intensities (calculated fully to allow independent visibility)
    i_core    = np.maximum((np.abs(a_core)**2) * 1e-8, 1e-16)
    i_in_peg  = np.maximum((np.abs(a_in_peg)**2) * 1e-8, 1e-16)
    i_pbo     = np.maximum((np.abs(a_pbo)**2) * 1e-8, 1e-16)
    i_out_peg = np.maximum((np.abs(a_out_peg)**2) * 1e-8, 1e-16)
    i_cross   = np.maximum(cross_term_intensity, 1e-16)
    
    return total_intensity, i_core, i_in_peg, i_pbo, i_out_peg, i_cross

# Setup Q-range
q = np.logspace(-3, -0.3, 300)

# Initial Parameters
init_sol_sld = 6.36
init_core_sld = 6.36  # Default to D2O
init_core_t = 40.0
init_in_peg_t = 15.0
init_pbo_t = 20.0
init_out_peg_t = 25.0
init_pbo_sld = 0.44
init_peg_sld = 0.64

# Create Plot (Wider figure to accommodate side-by-side plots)
fig = plt.figure(figsize=(13, 8))

# --- Main Scattering Plot ---
ax = fig.add_axes([0.07, 0.45, 0.55, 0.50])
i_tot, i_c, i_in_p, i_b, i_out_p, i_cross = calculate_components(q, init_sol_sld, init_core_t, init_in_peg_t, init_pbo_t, init_out_peg_t,
                                                                 init_core_sld, init_pbo_sld, init_peg_sld)

# Plotting (Using colorblind-friendly Okabe-Ito palette)
line_tot,     = ax.loglog(q, i_tot, lw=3, color='#000000', label='Total Model') # Black
line_core,    = ax.loglog(q, i_c, lw=1.5, color='#56B4E9', alpha=0.8, ls='--', label='Core Contrib.') # Sky Blue
line_in_peg,  = ax.loglog(q, i_in_p, lw=1.5, color='#009E73', alpha=0.8, ls='--', label='In-PEG Contrib.') # Bluish Green
line_pbo,     = ax.loglog(q, i_b, lw=1.5, color='#E69F00', alpha=0.8, ls='--', label='PBO Contrib.') # Orange
line_out_peg, = ax.loglog(q, i_out_p, lw=1.5, color='#CC79A7', alpha=0.8, ls='--', label='Out-PEG Contrib.') # Reddish Purple
line_cross,   = ax.loglog(q, i_cross, lw=1.5, color='#555555', alpha=0.8, ls=':', label='|Net Cross Terms|', visible=False) # Dark Grey

ax.set_xlabel('Momentum Transfer q ($\AA^{-1}$)')
ax.set_ylabel('Intensity I(q) ($cm^{-1}$)')
ax.set_title('SANS Vesicle Explorer (PEO-PBO-PEO)', pad=10)
ax.grid(True, which="both", ls="-", alpha=0.2)
ax.legend(loc='lower left', fontsize='small')
# Tick configuration: inward facing, on all four sides
ax.tick_params(direction='in', top=True, bottom=True, left=True, right=True, which='both')

# --- SLD Profile Plot (Side-by-side) ---
ax_prof = fig.add_axes([0.70, 0.45, 0.27, 0.50], facecolor='whitesmoke')
r_prof, sld_prof = get_sld_profile(init_core_t, init_in_peg_t, init_pbo_t, init_out_peg_t,
                                   init_core_sld, init_pbo_sld, init_peg_sld, init_sol_sld)
line_prof, = ax_prof.plot(r_prof, sld_prof, color='purple', lw=2.5)
ax_prof.set_title('Radial SLD Profile $\\rho(r)$', fontsize=11, pad=10)
ax_prof.set_xlabel('Radius $r$ (Å)', fontsize=10)
ax_prof.set_ylabel('SLD (10⁻⁶ Å⁻²)', fontsize=10)
ax_prof.grid(True, ls='-', alpha=0.4)
ax_prof.set_xlim(0, r_prof[-1])
ax_prof.set_ylim(min(sld_prof) - 1, max(sld_prof) + 1)
# Tick configuration for profile
ax_prof.tick_params(direction='in', top=True, bottom=True, left=True, right=True, which='both')

# UI Parameters Setup
axcolor = 'lightgoldenrodyellow'

def create_param(y_pos, label, min_val, max_val, init_val):
    """Helper function to create a synced Slider and TextBox."""
    ax_slider = plt.axes([0.48, y_pos, 0.35, 0.025], facecolor=axcolor)
    ax_text = plt.axes([0.85, y_pos, 0.06, 0.025])
    
    # Turn off slider's built-in value text safely
    slider = Slider(ax_slider, label, min_val, max_val, valinit=init_val)
    slider.valtext.set_visible(False)
    
    text_box = TextBox(ax_text, '', initial=f"{init_val:.2f}")
    
    def on_submit(text, s=slider, t=text_box):
        try:
            val = float(text)
            val = np.clip(val, min_val, max_val) # Prevent going out of bounds
            s.set_val(val)
            t.set_val(f"{val:.2f}")
        except ValueError:
            t.set_val(f"{s.val:.2f}") # Revert if input is invalid
            
    text_box.on_submit(on_submit)
    
    def on_change(val, t=text_box):
        if t.text != f"{val:.2f}":
            t.set_val(f"{val:.2f}")
            
    slider.on_changed(on_change)
    return slider, text_box

# Create the interactive UI elements (Spread vertically, aligned to the right)
s_sol, t_sol             = create_param(0.36, 'Solvent SLD (Out)', -0.5, 7.0, init_sol_sld)
s_core_sld, t_core_sld   = create_param(0.30, 'Core SLD (In)', -0.5, 7.0, init_core_sld)
s_core_t, t_core_t       = create_param(0.24, 'Core Radius (Å)', 5.0, 200.0, init_core_t)
s_in_peg_t, t_in_peg_t   = create_param(0.18, 'Inner PEG Thick (Å)', 0.0, 150.0, init_in_peg_t)
s_pbo_t, t_pbo_t         = create_param(0.12, 'PBO Thick (Å)', 0.0, 150.0, init_pbo_t)
s_out_peg_t, t_out_peg_t = create_param(0.06, 'Outer PEG Thick (Å)', 0.0, 150.0, init_out_peg_t)

# --- Compact Dual Control CheckButtons ---
# Expanded background box
ax_bg = plt.axes([0.02, 0.05, 0.20, 0.33], facecolor=axcolor)
ax_bg.set_xticks([])
ax_bg.set_yticks([])

# 1. Visibility Column (Just boxes, overlapping transparent axes trick)
ax_vis = plt.axes([0.03, 0.06, 0.08, 0.30], frameon=False)
check_vis = CheckButtons(ax_vis, ('', '', '', '', ''), (True, True, True, True, False))

# 2. Math Inclusion Column (Boxes + Labels, shifted right)
ax_inc = plt.axes([0.08, 0.06, 0.12, 0.30], frameon=False)
check_inc = CheckButtons(ax_inc, ('Core', 'In-PEG', 'PBO', 'Out-PEG', 'Cross'), (True, True, True, True, True))

# Headers for the compact checkboxes
fig.text(0.050, 0.39, 'Vis', ha='center', va='bottom', fontweight='bold', fontsize=9)
fig.text(0.120, 0.39, 'Math', ha='center', va='bottom', fontweight='bold', fontsize=9)

# --- Reference SLDs Text Box ---
info_text = f"Fixed SLDs (10⁻⁶ Å⁻²):\n  PBO: {init_pbo_sld}\n  PEO/PEG: {init_peg_sld}"
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
fig.text(0.25, 0.21, info_text, fontsize=10, verticalalignment='center', bbox=props)


def update(val=None):
    # Get toggle states
    flags_vis = check_vis.get_status()
    flags_inc = check_inc.get_status()
    
    # Recalculate everything with new slider and math inclusion values
    res_tot, res_c, res_in_p, res_b, res_out_p, res_cross = calculate_components(
        q, s_sol.val, s_core_t.val, s_in_peg_t.val, s_pbo_t.val, s_out_peg_t.val,
        s_core_sld.val, init_pbo_sld, init_peg_sld,
        include_core=flags_inc[0], include_in_peg=flags_inc[1], include_pbo=flags_inc[2], include_out_peg=flags_inc[3],
        use_cross_terms=flags_inc[4]
    )
    
    # Update data for main plot lines
    line_tot.set_ydata(res_tot)
    line_core.set_ydata(res_c)
    line_in_peg.set_ydata(res_in_p)
    line_pbo.set_ydata(res_b)
    line_out_peg.set_ydata(res_out_p)
    line_cross.set_ydata(res_cross)
    
    # Toggle individual dashed line visibility independently
    line_core.set_visible(flags_vis[0])
    line_in_peg.set_visible(flags_vis[1])
    line_pbo.set_visible(flags_vis[2])
    line_out_peg.set_visible(flags_vis[3])
    line_cross.set_visible(flags_vis[4])
    
    # Update main axes limit dynamically based on active lines
    active_data = [res_tot]
    if flags_vis[0]: active_data.append(res_c)
    if flags_vis[1]: active_data.append(res_in_p)
    if flags_vis[2]: active_data.append(res_b)
    if flags_vis[3]: active_data.append(res_out_p)
    if flags_vis[4]: active_data.append(res_cross)
    
    # Smart filtering for minimum limits so it doesn't plunge to 1e-16 unless truly flat
    active_maxes = [arr.max() for arr in active_data if arr.max() > 1e-15]
    active_mins = [arr[arr > 1e-15].min() for arr in active_data if np.any(arr > 1e-15)]
    
    if active_maxes and active_mins:
        max_val = max(active_maxes)
        min_val = min(active_mins)
        ax.set_ylim(min_val * 0.5, max_val * 5)
    else:
        # Fallback if literally everything is zeroed out
        ax.set_ylim(1e-12, 1e-6)
    
    # --- Update SLD Profile Plot ---
    new_r, new_sld = get_sld_profile(s_core_t.val, s_in_peg_t.val, s_pbo_t.val, s_out_peg_t.val,
                                     s_core_sld.val, init_pbo_sld, init_peg_sld, s_sol.val)
    line_prof.set_data(new_r, new_sld)
    ax_prof.set_xlim(0, new_r[-1])
    ax_prof.set_ylim(min(new_sld) - 1, max(new_sld) + 1)
    
    fig.canvas.draw_idle()

# Bind the update function to all UI elements
s_sol.on_changed(update)
s_core_sld.on_changed(update)
s_core_t.on_changed(update)
s_in_peg_t.on_changed(update)
s_pbo_t.on_changed(update)
s_out_peg_t.on_changed(update)
check_vis.on_clicked(update)
check_inc.on_clicked(update)

# --- Save & Reset Buttons ---

# Save SVG Button
ax_save_svg = plt.axes([0.65, 0.01, 0.09, 0.04])
btn_svg = Button(ax_save_svg, 'Save SVG', color=axcolor, hovercolor='0.975')

def save_svg_fig(event):
    fig.savefig('sans_vesicle_explorer.svg', format='svg', bbox_inches='tight')
    print("Saved SANS profile as 'sans_vesicle_explorer.svg'")

btn_svg.on_clicked(save_svg_fig)

# Save PDF Button
ax_save_pdf = plt.axes([0.75, 0.01, 0.09, 0.04])
btn_pdf = Button(ax_save_pdf, 'Save PDF', color=axcolor, hovercolor='0.975')

def save_pdf_fig(event):
    fig.savefig('sans_vesicle_explorer.pdf', format='pdf', bbox_inches='tight')
    print("Saved SANS profile as 'sans_vesicle_explorer.pdf'")

btn_pdf.on_clicked(save_pdf_fig)

# Reset Button
resetax = plt.axes([0.85, 0.01, 0.09, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    s_sol.reset()
    s_core_sld.reset()
    s_core_t.reset()
    s_in_peg_t.reset()
    s_pbo_t.reset()
    s_out_peg_t.reset()
    
    # Reset checkboxes back to default state
    for check_group, defaults in [(check_vis, [True, True, True, True, False]),
                                  (check_inc, [True, True, True, True, True])]:
        for i, status in enumerate(check_group.get_status()):
            if status != defaults[i]:
                check_group.set_active(i)

button.on_clicked(reset)

plt.show()
