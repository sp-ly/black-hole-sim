import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from typing import cast, Any

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

G = 6.674e-11
c = 3e8
M_sun = 1.989e30

class SchwarzschildBlackHole:

    def __init__(self, mass_in_solar_masses):
        self.mass = mass_in_solar_masses * M_sun
        self.rs_physical = (2 * G * self.mass) / (c ** 2)
        self.scale = self.rs_physical / 2
        self.M = 1
        self.rs = 2
        self.rs_3 = 3 * self.scale

    def photon_sphere_radius(self):
        return self.rs_3

class GeodesicParticle:

    def __init__(self, r, theta, phi, dr_dtau, dphi_dtau, E, L):
        self.r = r
        self.theta = theta
        self.phi = phi
        self.dr_dtau = dr_dtau
        self.E = E
        self.L = L
        self.dphi_dtau = self.L / self.r**2 if self.r > 0 else 0
        self.trajectory = [(r, phi)]
        self.captured = False
        self.crossed_horizon = False

    @classmethod
    def from_initial_conditions(cls, r0, phi0, vr0, vphi0, black_hole):
        r_geom = r0 / black_hole.scale
        vr_geom = vr0 / c
        vphi_geom = vphi0 / c
        L = r_geom * vphi_geom
        E = np.sqrt(vr_geom**2 + (1 - 2 / r_geom) * (1 + vphi_geom**2))
        dr_dtau = vr_geom
        return cls(r_geom, np.pi/2, phi0, dr_dtau, 0, E, L)

    def effective_potential(self, r, black_hole):
        if r <= 2 * black_hole.M:
            return float('inf')
        return (1 - 2*black_hole.M/r) * (1 + self.L**2 / r**2)

    def update(self, black_hole, dtau):
        M = black_hole.M
        r = self.r
        if r <= 2 * M:
            self.crossed_horizon = True
            self.captured = True
            return
        V_eff = self.effective_potential(r, black_hole)
        if V_eff > self.E**2:
            self.dr_dtau = -self.dr_dtau
        dV_dr = (-2 * self.L**2 / r**3 + 2 * M / r**2 + 6 * M * self.L**2 / r**4)
        if abs(self.dr_dtau) > 1e-10:
            ddr_dtau = dV_dr / (2 * self.dr_dtau)
        else:
            ddr_dtau = 0
        self.dr_dtau += ddr_dtau * dtau
        self.r += self.dr_dtau * dtau
        self.dphi_dtau = self.L / self.r**2 if self.r > 0 else 0
        self.phi += self.dphi_dtau * dtau
        r_physical = self.r * black_hole.scale
        self.trajectory.append((r_physical, self.phi))
        if len(self.trajectory) > 500:
            self.trajectory.pop(0)
        if self.r <= 2 * M or len(self.trajectory) > 10000:
            self.captured = True

def create_sphere(radius, resolution=8):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def visualize_schwarzschild_particle(black_hole, particle, num_steps=500, dtau=0.1):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    rs_physical = black_hole.rs_physical
    r_photon_physical = black_hole.photon_sphere_radius()
    x_horizon, y_horizon, z_horizon = create_sphere(rs_physical, resolution=8)
    x_photon, y_photon, z_photon = create_sphere(r_photon_physical, resolution=6)
    ax.plot_surface(x_horizon, y_horizon, z_horizon, color='black', alpha=0.95,
                    edgecolor='darkred', linewidth=0.1, shade=True)
    ax.plot_wireframe(x_photon, y_photon, z_photon, color='#ffb7c5',
                      alpha=0.6, linewidth=0.6, rstride=2, cstride=2)

    #particle scatter plot
    r_init = particle.r * black_hole.scale
    x_init = r_init * np.cos(particle.phi)
    y_init = r_init * np.sin(particle.phi)
    z_init = 0

    particle_scatter = ax.scatter(cast(Any, [x_init]), cast(Any, [y_init]), cast(Any, [z_init]),
                                 s=64, c='cyan', alpha=0.9, marker='o')

    trajectory_line, = ax.plot([], [], [], '-', color='cyan',
                              linewidth=1.5, alpha=0.7)

    #axes
    ax.set_xlabel('x', color='white', fontsize=16, labelpad=15)
    ax.set_ylabel('y', color='white', fontsize=16, labelpad=15)
    ax.set_zlabel('z', color='white', fontsize=16, labelpad=15)
    cast(Any, ax).set_xticklabels([])  # type: ignore
    cast(Any, ax).set_yticklabels([])  # type: ignore
    cast(Any, ax).set_zticklabels([])  # type: ignore

    ax.grid(True, color='white', alpha=0.3)
    ax.xaxis.set_tick_params(colors='white')
    ax.yaxis.set_tick_params(colors='white')
    ax.zaxis.set_tick_params(colors='white')
    cast(Any, ax.xaxis).set_pane_color((0, 0, 0, 0))
    cast(Any, ax.yaxis).set_pane_color((0, 0, 0, 0))
    cast(Any, ax.zaxis).set_pane_color((0, 0, 0, 0))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    max_range = r_photon_physical * 2
    ax.set_xlim(cast(Any, (-max_range, max_range)))
    ax.set_ylim(cast(Any, (-max_range, max_range)))
    ax.set_zlim(cast(Any, (-max_range, max_range)))

    title_text = r'$\mathbf{Schwarzschild\ Black\ Hole}$'
    ax.text2D(-0.25, 0.97, title_text, transform=ax.transAxes,
              fontsize=18, color='white', verticalalignment='bottom')

    mass_str = f'{black_hole.mass/M_sun:.2e}'
    rs_str = f'{black_hole.rs_physical:.2e}'
    r_photon_str = f'{r_photon_physical:.2e}'

    info_text = (
        r'$\mathbf{Mass:}\ ' + mass_str + r'\ M_{\odot}$' + '\n' +
        r'$\mathbf{Event\ Horizon:}\ r_s = ' + rs_str + r'\ \mathrm{m}$' + '\n' +
        r'$\mathbf{Photon\ Sphere:}\ 3M = ' + r_photon_str + r'\ \mathrm{m}$')

    ax.text2D(-0.25, 0.95, info_text, transform=ax.transAxes,
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='black', alpha=0.85, edgecolor='#ffb7c5'),
              color='white')

    #particle status
    particle_status = ax.text2D(0.02, 0.25, '', transform=ax.transAxes,
                                fontsize=9, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='black',
                                        alpha=0.85, edgecolor='cyan'),
                                color='white')

    #animation state
    anim_state = {'step': 0, 'paused': False, 'speed': 1.0, 'started': False}

    #initial conditions
    initial_r = particle.r
    initial_phi = particle.phi
    initial_dr_dtau = particle.dr_dtau
    initial_dphi_dtau = particle.dphi_dtau

    def animate(frame):
        nonlocal particle_scatter

        if not anim_state['started']:
            #initial
            r_phys = particle.r * black_hole.rs / (2 * black_hole.M)
            status = "Ready"
            status_text = (f"{status}\n"
                          f"Radius: {r_phys/black_hole.rs:.3f} $r_s$\n"
                          f"Energy: {particle.E:.4f}\n"
                          f"Ang. Mom.: {particle.L:.4f}\n"
                          f"Step: {anim_state['step']}/{num_steps}")
            particle_status.set_text(status_text)
            return particle_scatter, trajectory_line

        if not anim_state['paused'] and anim_state['step'] < num_steps and not particle.captured:
            #update particle
            updates_per_frame = max(1, int(anim_state['speed'] * 5))
            for _ in range(updates_per_frame):
                if not particle.captured:
                    particle.update(black_hole, dtau)

            anim_state['step'] += 1

            #update particle position
            r_phys = particle.r * black_hole.rs / (2 * black_hole.M)
            x = r_phys * np.cos(particle.phi)
            y = r_phys * np.sin(particle.phi)
            z = 0

            try:
                particle_scatter.remove()
            except (ValueError, AttributeError):
                pass
            particle_scatter = ax.scatter(cast(Any, [x]), cast(Any, [y]), cast(Any, [z]),
                                         s=64, c='cyan', alpha=0.9, marker='o')

            #update trajectory
            if len(particle.trajectory) > 1:
                traj_r, traj_phi = zip(*particle.trajectory[-500:])
                traj_x = np.array(traj_r) * np.cos(np.array(traj_phi))
                traj_y = np.array(traj_r) * np.sin(np.array(traj_phi))
                traj_z = np.zeros_like(traj_x)
                cast(Any, trajectory_line).set_data_3d(traj_x, traj_y, traj_z)

        #update status
        r_phys = particle.r * black_hole.rs / (2 * black_hole.M)
        if particle.crossed_horizon:
            status = "Event Horizon Crossed"
        elif particle.captured:
            status = "Captured"
        else:
            status = "Active"

        status_text = (f"{status}\n"
                      f"Radius: {r_phys/black_hole.rs:.3f} $r_s$\n"
                      f"Energy: {particle.E:.4f}\n"
                      f"Ang. Mom.: {particle.L:.4f}\n"
                      f"Step: {anim_state['step']}/{num_steps}")
        particle_status.set_text(status_text)

        return particle_scatter, trajectory_line

    #animation
    anim = FuncAnimation(fig, animate, frames=num_steps,
                        interval=50, blit=False, repeat=True)

    # controls
    ax_start = plt.axes(cast(Any, [0.81, 0.20, 0.1, 0.04]))
    btn_start = Button(ax_start, 'Start', color='green', hovercolor='lightgreen')

    ax_reset = plt.axes(cast(Any, [0.81, 0.15, 0.1, 0.04]))
    btn_reset = Button(ax_reset, 'Reset', color='gray', hovercolor='lightgray')

    ax_pause = plt.axes(cast(Any, [0.81, 0.10, 0.1, 0.04]))
    btn_pause = Button(ax_pause, 'Pause', color='gray', hovercolor='lightgray')

    ax_speed = plt.axes(cast(Any, [0.81, 0.05, 0.1, 0.03]))
    slider_speed = Slider(ax_speed, 'Speed', 0.1, 3.0, valinit=1.0, color='cyan',
                         valstep=0.1)

    fig.text(0.805, 0.038, '0.1x', fontsize=8, color='white')
    fig.text(0.875, 0.038, '1.5x', fontsize=8, color='white')
    fig.text(0.915, 0.038, '3x', fontsize=8, color='white')

    def start(event):
        if anim_state['started']:
            return
        anim_state['started'] = True
        btn_start.label.set_text('Running')
        btn_start.color = 'gray'
        btn_start.hovercolor = 'lightgray'

    def reset(event):
        nonlocal particle_scatter, anim
        #reset particle
        particle.r = initial_r
        particle.phi = initial_phi
        particle.dr_dtau = initial_dr_dtau
        particle.dphi_dtau = initial_dphi_dtau
        particle.trajectory = [(particle.r * black_hole.rs / (2 * black_hole.M), particle.phi)]
        particle.captured = False
        particle.crossed_horizon = False

        anim_state['step'] = 0
        anim_state['paused'] = False
        anim_state['started'] = False
        anim_state['speed'] = 1.0

        cast(Any, trajectory_line).set_data_3d([], [], [])

        #reset scatter
        r_phys = particle.r * black_hole.rs / (2 * black_hole.M)
        x = r_phys * np.cos(particle.phi)
        y = r_phys * np.sin(particle.phi)
        z = 0
        try:
            particle_scatter.remove()
        except (ValueError, AttributeError):
            pass
        particle_scatter = ax.scatter(cast(Any, [x]), cast(Any, [y]), cast(Any, [z]),
                                     s=64, c='cyan', alpha=0.9, marker='o')

        #reset buttons
        btn_start.label.set_text('Start')
        btn_start.color = 'green'
        btn_start.hovercolor = 'lightgreen'
        btn_pause.label.set_text('Pause')
        slider_speed.set_val(1.0)

        #restart animation
        anim.event_source.stop()
        anim = FuncAnimation(fig, animate, frames=num_steps,
                            interval=50, blit=False, repeat=True)

    def pause(event):
        anim_state['paused'] = not anim_state['paused']
        btn_pause.label.set_text('Resume' if anim_state['paused'] else 'Pause')

    def update_speed(val):
        anim_state['speed'] = val

    btn_start.on_clicked(start)
    btn_reset.on_clicked(reset)
    btn_pause.on_clicked(pause)
    slider_speed.on_changed(update_speed)

    plt.show()

if __name__ == "__main__":
    black_hole = SchwarzschildBlackHole(mass_in_solar_masses=4.297e6)
    r_start_geom = 8
    r_start_physical = r_start_geom * black_hole.scale
    v_circ = c * np.sqrt(black_hole.M / r_start_geom)
    v_phi = v_circ
    v_r = -c * 0.01
    particle = GeodesicParticle.from_initial_conditions(
        r_start_physical, 0, v_r, v_phi, black_hole
    )
    visualize_schwarzschild_particle(black_hole, particle, num_steps=1000, dtau=0.05)