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

class BlackHole:
    def __init__(self, mass_in_solar_masses, is_rotating=False, spin_parameter=0.0):
        self.mass = mass_in_solar_masses * M_sun
        self.is_rotating = is_rotating
        self.spin_parameter = spin_parameter

        self.schwarzschild_radius = (2 * G * self.mass) / (c ** 2)
        
        self.rs_1_5 = self.schwarzschild_radius * 1.5
        self.rs_0_00005 = self.schwarzschild_radius * 5e-5
        self.GM = G * self.mass
    
    def gravitational_acceleration(self, position):
        r = np.linalg.norm(position)
        if r < 1e-10:
            return np.array([0.0, 0.0, 0.0])
        
        direction = -position / r
        magnitude = self.GM / (r ** 2)
        
        return np.multiply(magnitude, direction)

class Particle:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.trajectory = [self.position.copy()]
        self.captured = False
        self.in_photon_sphere = False
        self.crossed_event_horizon = False
        self.burst = None
        
    def update(self, black_hole, dt):
        rs = black_hole.schwarzschild_radius
        rs_1_5 = black_hole.rs_1_5

        distance = np.linalg.norm(self.position)

        substeps = 1 if distance > rs_1_5 else max(1, min(4, int(rs / max(distance, 1e-6))))
        dt_sub = dt / substeps

        for _ in range(substeps):
            #flags
            if distance <= rs_1_5:
                self.in_photon_sphere = True
            if distance <= rs and not self.crossed_event_horizon:
                self.crossed_event_horizon = True

            #velocity-verlet
            a0 = black_hole.gravitational_acceleration(self.position)
            v_half = self.velocity + 0.5 * a0 * dt_sub
            pos_new = self.position + v_half * dt_sub

            #capture check 1
            if np.linalg.norm(pos_new) <= 1e-12:
                self.position[:] = 0.0
                self.velocity[:] = 0.0
                self.captured = True
                break

            a1 = black_hole.gravitational_acceleration(pos_new)
            vel_new = v_half + 0.5 * a1 * dt_sub

            #velo decomp
            pos_norm = np.linalg.norm(pos_new)
            if pos_norm > 1e-12:
                dir_to_center = -pos_new / pos_norm
                v_radial = np.dot(vel_new, dir_to_center)
                v_tangential = vel_new - v_radial * dir_to_center

                #damping
                if pos_norm > rs_1_5:
                    #outside ps
                    damping = 1.0 - 4e-5 * dt_sub
                    v_radial *= damping
                    v_tangential *= damping
                else:
                    #inside ps
                    proximity = (rs_1_5 - pos_norm) / rs_1_5
                    v_radial -= (0.02 + 0.08 * proximity) * np.linalg.norm(vel_new)
                    v_tangential *= max(0.1, 1.0 - (5e-4 + 2e-3 * proximity) * dt_sub)

                if self.crossed_event_horizon:
                    v_radial = v_radial * 1.30 - 0.08 * np.linalg.norm(vel_new)
                    v_tangential *= 0.5

                self.velocity = dir_to_center * v_radial + v_tangential

            self.position = pos_new
            distance = pos_norm

            #capture check 2
            if distance <= max(1e-4, black_hole.rs_0_00005):
                self.captured = True
                # burst creation
                Nfrag = 8
                base_speed = max(np.linalg.norm(self.velocity) * 0.6, 1.0)
                dirs = np.random.normal(size=(Nfrag, 3))
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                speeds = base_speed * (0.5 + np.random.rand(Nfrag) * 1.5)
                frags_vel = dirs * speeds[:, None]
                self.burst = {'pos': np.zeros_like(frags_vel), 'vel': frags_vel, 'life': 60}
                self.position[:] = 0.0
                self.velocity[:] = 0.0
                break

        #trajectory length
        if len(self.trajectory) > 500:  # Keep only recent points
            self.trajectory = self.trajectory[-250:]  # Keep half

        self.trajectory.append(self.position.copy())
    
    def reset(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.trajectory = [self.position.copy()]
        self.captured = False
        self.in_photon_sphere = False
        self.crossed_event_horizon = False
        self.burst = None

def create_sphere(radius, resolution=15):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    return x, y, z

def visualize_with_particle(black_hole, particle, num_steps=300, dt=100):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax: Axes3D = ax  # type: ignore
    
    x, y, z = create_sphere(black_hole.schwarzschild_radius, resolution=12)
    x_glow, y_glow, z_glow = create_sphere(black_hole.schwarzschild_radius * 1.5, resolution=8)
    pink = '#ffb7c5'
    
    #plot black hole
    ax.plot_surface(x, y, z, color='black', alpha=0.95, 
                    edgecolor='darkred', linewidth=0.1, shade=True)
    
    ax.plot_wireframe(x_glow, y_glow, z_glow, color=pink,
                      alpha=0.6, linewidth=0.6, rstride=2, cstride=2)
    
    particle_scatter = ax.scatter(cast(Any, [particle.position[0]]), 
                                 cast(Any, [particle.position[1]]), 
                                 cast(Any, [particle.position[2]]), 
                                 s=64, c='cyan', alpha=0.9, marker='o')

    burst_scatter = ax.scatter(cast(Any, []), cast(Any, []), cast(Any, []), s=30, c='orange', alpha=0.9)
    
    trajectory_line, = ax.plot([], [], [], '-', color='cyan', 
                              linewidth=1.5, alpha=0.7)
    
    #axes
    ax.set_xlabel('x', color='white', fontsize=16, labelpad=15)
    ax.set_ylabel('y', color='white', fontsize=16, labelpad=15)
    ax.set_zlabel('z', color='white', fontsize=16, labelpad=15)
    ax.set_xticklabels(cast(Any, []))  # type: ignore
    ax.set_yticklabels(cast(Any, []))  # type: ignore
    ax.set_zticklabels(cast(Any, []))  # type: ignore
    
    ax.grid(True, color='white', alpha=0.3)
    #coordinates
    ax.xaxis.set_tick_params(colors='white')
    ax.yaxis.set_tick_params(colors='white')
    ax.zaxis.set_tick_params(colors='white')
    ax.xaxis.set_pane_color((0, 0, 0, 0))  # type: ignore
    ax.yaxis.set_pane_color((0, 0, 0, 0))  # type: ignore
    ax.zaxis.set_pane_color((0, 0, 0, 0))  # type: ignore
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    max_range = black_hole.schwarzschild_radius * 3
    ax.set_xlim(cast(Any, (-max_range, max_range)))
    ax.set_ylim(cast(Any, (-max_range, max_range)))
    ax.set_zlim(cast(Any, (-max_range, max_range)))
    
    title_text = r'$\mathbf{Sagittarius\ A^*\ (Kerr\ Model)}$'
    ax.text2D(-0.25, 0.97, title_text, transform=ax.transAxes,
              fontsize=18, color='white', verticalalignment='bottom')

    info_text = (
        r'$\mathbf{Mass:}\ \sim 4.297 \times 10^6\ M_{\odot}$' + '\n' +
        r'$\mathbf{Event\ Horizon\ Radius}^*\ \mathrm{(Red\ ring):}\ r_{+} = 9.115 \times 10^9\ \mathrm{m}$' + '\n' +
        r'$\mathbf{Photon\ Sphere\ Radius}^*\ \mathrm{(Pink\ ring):}$' + '\n' +
        r'$r_{\mathrm{pro}} = 9.865 \times 10^9\ \mathrm{m};\ r_{\mathrm{ret}} = 24.814 \times 10^9\ \mathrm{m}$')
    
    ax.text2D(-0.25, 0.95, info_text, transform=ax.transAxes,
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='black', alpha=0.85, edgecolor=pink),
              color='white')
    
    #particle status
    particle_status = ax.text2D(0.02, 0.25, '', transform=ax.transAxes,
                                fontsize=9, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='black', 
                                        alpha=0.85, edgecolor='cyan'),
                                color='white')
    
    #animation state
    anim_state = {'step': 0, 'paused': False, 'speed': 0.25, 'started': False}
    
    #initial conditions
    initial_pos = particle.position.copy()
    initial_vel = particle.velocity.copy()
    
    def animate(frame):
        nonlocal particle_scatter, burst_scatter
        if not anim_state['started']:
            #update status
            distance = np.linalg.norm(particle.position)
            speed = np.linalg.norm(particle.velocity)
            r_s = black_hole.schwarzschild_radius
            status = "Status: Ready"
            status_text = (f"{status}\n"
                          f"Distance: {distance/r_s:.3f} $r_s$\n"
                          f"Speed: {speed/c:.4f} $c$\n"
                          f"Step: {anim_state['step']}/{num_steps}")
            particle_status.set_text(status_text)
            return particle_scatter, trajectory_line, burst_scatter
        
        if not anim_state['paused'] and anim_state['step'] < num_steps:
            #physics update
            updates_per_frame = max(1, int(anim_state['speed'] * 12))
            for _ in range(updates_per_frame):
                if not particle.captured:
                    particle.update(black_hole, dt)
                else:
                    break
            
            anim_state['step'] += 1
            
            #particle scatter
            try:
                particle_scatter.remove()
            except (ValueError, AttributeError):
                pass  # Already removed
            particle_scatter = ax.scatter(cast(Any, [particle.position[0]]), 
                                         cast(Any, [particle.position[1]]), 
                                         cast(Any, [particle.position[2]]), 
                                         s=64, c='cyan', alpha=0.9, marker='o')
            
            traj = np.array(particle.trajectory)
            trajectory_line.set_data_3d(traj[:, 0], traj[:, 1], traj[:, 2])  # type: ignore
        
        #rotation
        if black_hole.is_rotating:
            # Event horizon rotation (frame dragging effect)
            theta_horizon = np.radians(frame * black_hole.spin_parameter * 1.5)
            rotation_matrix_horizon = np.array([
                [np.cos(theta_horizon), -np.sin(theta_horizon), 0],
                [np.sin(theta_horizon), np.cos(theta_horizon), 0],
                [0, 0, 1]
            ])
            
            #ps rotation
            theta_photon = np.radians(frame * black_hole.spin_parameter * -2.5)
            rotation_matrix_photon = np.array([
                [np.cos(theta_photon), -np.sin(theta_photon), 0],
                [np.sin(theta_photon), np.cos(theta_photon), 0],
                [0, 0, 1]
            ])
            
            #event horizon
            points = np.array([x.flatten(), y.flatten(), z.flatten()])
            rotated_points = np.dot(rotation_matrix_horizon, points)
            x_rot = rotated_points[0].reshape(x.shape)
            y_rot = rotated_points[1].reshape(y.shape)
            z_rot = rotated_points[2].reshape(z.shape)
            
            #ps
            points_glow = np.array([x_glow.flatten(), y_glow.flatten(), z_glow.flatten()])
            rotated_points_glow = np.dot(rotation_matrix_photon, points_glow)
            x_glow_rot = rotated_points_glow[0].reshape(x_glow.shape)
            y_glow_rot = rotated_points_glow[1].reshape(y_glow.shape)
            z_glow_rot = rotated_points_glow[2].reshape(z_glow.shape)
            
            while len(ax.collections) > 2:
                ax.collections[0].remove() #old
            
            #new
            ax.plot_surface(x_rot, y_rot, z_rot, 
                          color='black', alpha=0.95,
                          edgecolor='darkred', linewidth=0.1, shade=True)
            ax.plot_wireframe(x_glow_rot, y_glow_rot, z_glow_rot,
                            color=pink, alpha=0.6, linewidth=0.6,
                            rstride=2, cstride=2)
        
        #status text
        if not particle.captured:
            distance = np.linalg.norm(particle.position)
            speed = np.linalg.norm(particle.velocity)
            r_s = black_hole.schwarzschild_radius
            
            if particle.crossed_event_horizon:
                status = "Status: Event Horizon"
                extra_info = "\nSpiraling inward"
            elif particle.in_photon_sphere:
                status = "Status: Photon Sphere"
                extra_info = ""
            else:
                status = "Status: Active"
                extra_info = ""
            
            status_text = (f"{status}{extra_info}\n"
                          f"Distance: {distance/r_s:.3f} $r_s$\n"
                          f"Speed: {speed/c:.4f} $c$\n"
                          f"Step: {anim_state['step']}/{num_steps}")
            particle_status.set_text(status_text)
        else:
            if particle.captured:
                status_text = (f"Status: SINGULARITY\n"
                               f"Fragments: {0 if particle.burst is None else particle.burst['pos'].shape[0]}\n"
                               f"Life: {0 if particle.burst is None else particle.burst['life']}")
                particle_status.set_text(status_text)

        #burst animation
        if particle.burst is not None and particle.burst['life'] > 0:
            frame_dt = dt * max(1, int(anim_state['speed'] * 12))
            particle.burst['pos'] += particle.burst['vel'] * (frame_dt * 0.02)  # scale for visual pace
            particle.burst['life'] -= 1
            pts = particle.burst['pos']
            try:
                burst_scatter.remove()
            except (ValueError, AttributeError):
                pass  # Already removed
            burst_scatter = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=30, c='orange', alpha=max(0.0, particle.burst['life'] / 90.0))
            if particle.burst['life'] <= 0:
                particle.burst = None
                try:
                    burst_scatter.remove()
                except (ValueError, AttributeError):
                    pass  # Already removed
                burst_scatter = ax.scatter(cast(Any, []), cast(Any, []), cast(Any, []), s=30, c='orange', alpha=0.0)
         
        return particle_scatter, trajectory_line, burst_scatter
    
    #initial animation
    anim = FuncAnimation(fig, animate, frames=num_steps,
                        interval=50, blit=False, repeat=True)
    
    ax_start = plt.axes(cast(Any, [0.81, 0.20, 0.1, 0.04]))
    btn_start = Button(ax_start, 'Start', color='green', hovercolor='lightgreen')
    
    ax_reset = plt.axes(cast(Any, [0.81, 0.15, 0.1, 0.04]))
    btn_reset = Button(ax_reset, 'Reset', color='gray', hovercolor='lightgray')
    
    ax_pause = plt.axes(cast(Any, [0.81, 0.10, 0.1, 0.04]))
    btn_pause = Button(ax_pause, 'Pause', color='gray', hovercolor='lightgray')
    
    ax_speed = plt.axes(cast(Any, [0.81, 0.05, 0.1, 0.03]))
    slider_speed = Slider(ax_speed, 'Speed', 0.0, 2.0, valinit=0.25, color='cyan',
                         valstep=0.05)
    
    fig.text(0.805, 0.038, '0%', fontsize=8, color='white')
    fig.text(0.875, 0.038, '100%', fontsize=8, color='white')
    fig.text(0.915, 0.038, '200%', fontsize=8, color='white')
    
    def start(event):
        if anim_state['started']:
            return
        anim_state['started'] = True
        btn_start.label.set_text('Running')
        btn_start.color = 'gray'
        btn_start.hovercolor = 'lightgray'
    
    def reset(event):
        nonlocal particle_scatter, burst_scatter, anim
        particle.reset(initial_pos, initial_vel)
        anim_state['step'] = 0
        anim_state['paused'] = False
        anim_state['started'] = False
        anim_state['speed'] = 0.25
        trajectory_line.set_data_3d([], [], [])  # type: ignore
        #scatter plot
        try:
            particle_scatter.remove()
        except (ValueError, AttributeError):
            pass  # Already removed or doesn't exist
        particle_scatter = ax.scatter(cast(Any, [particle.position[0]]), 
                                     cast(Any, [particle.position[1]]), 
                                     cast(Any, [particle.position[2]]), 
                                     s=64, c='cyan', alpha=0.9, marker='o')
        #reset burst
        try:
            burst_scatter.remove()
        except (ValueError, AttributeError):
            pass  # Already removed or doesn't exist
        burst_scatter = ax.scatter(cast(Any, []), cast(Any, []), cast(Any, []), s=30, c='orange', alpha=0.9)
        #reset button
        btn_start.label.set_text('Start')
        btn_start.color = 'green'
        btn_start.hovercolor = 'lightgreen'
        btn_pause.label.set_text('Pause')
        slider_speed.set_val(0.25)
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
    print("Black Hole Particle Simulator")
    print("=" * 50)
    
    sgr_a_star = BlackHole(mass_in_solar_masses=4.297e6, is_rotating=True, spin_parameter=0.9)
    
    #test particle
    r_start = sgr_a_star.schwarzschild_radius * 2.5
    v_orbital = np.sqrt(G * sgr_a_star.mass / r_start)

    particle = Particle(
        position=[r_start, 0, 0],
        velocity=[0, v_orbital * 0.90, 0],)

    visualize_with_particle(sgr_a_star, particle, num_steps=300, dt=20)