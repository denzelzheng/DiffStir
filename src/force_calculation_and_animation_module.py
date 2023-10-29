from src.simulator.simulation_functions import *

def force_calculation_and_animation(presentation_time: ti.f32):
    print()
    initialize_objects()
    print("Simulating the process for visualization...")
    loss[None] = 0
    for f in range(int(presentation_time / dt - 1)):
        forward(f)
    compute_loss()
    print("eta0:", constitutive_parameters[0])
    print("Loss:", loss[None])
    pos_series = x.to_numpy()
    operator_pos_series = operator_x.to_numpy()
    container_pos_series = container_x.to_numpy()
    total_operator_dv_series = total_operator_dv.to_numpy() \
                               * operator_p_vol * operator_density / dt
    force_x = total_operator_dv_series[:, 0]
    force_z = total_operator_dv_series[:, 2]
    np.save(os.path.abspath(os.path.dirname(__file__)) + "/../output/force_profile_x.npy", force_x)
    np.save(os.path.abspath(os.path.dirname(__file__)) + "/../output/force_profile_z.npy", force_z)
    counter = 0
    while True:
        counter += 1
        word = input("Finished. Press p + Enter to continue...")
        if word == 'p':
            break
    pcd = o3d.geometry.PointCloud()
    operator_pcd = o3d.geometry.PointCloud()
    container_pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    elevation = dx * (bound - 1)
    lower_bound = elevation
    upper_bound = 1 - elevation
    points = [
        [lower_bound, lower_bound, lower_bound],
        [upper_bound, lower_bound, lower_bound],
        [lower_bound, upper_bound, lower_bound],
        [upper_bound, upper_bound, lower_bound],
        [lower_bound, lower_bound, upper_bound],
        [upper_bound, lower_bound, upper_bound],
        [lower_bound, upper_bound, upper_bound],
        [upper_bound, upper_bound, upper_bound],
    ]
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6],
        [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    stick_center_series = np.mean(operator_pos_series, axis=1)
    np.save(os.path.abspath(os.path.dirname(__file__)) + "/../output/liquid_point_clouds.npy", pos_series)
    np.save(os.path.abspath(os.path.dirname(__file__)) + "/../output/stick_positions.npy", stick_center_series[:-1,:])
    for j in range(f):
        play_rate = 11
        if (j % play_rate == 0):
            pcd.points = o3d.utility.Vector3dVector(pos_series[j])
            operator_pcd.points = o3d.utility.Vector3dVector(operator_pos_series[j])
            operator_pcd.paint_uniform_color([x / 255 for x in [227, 168, 105]])
            operator_pcd, _ = operator_pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=37.0)
            container_pcd.points = o3d.utility.Vector3dVector(container_pos_series[j])
            container_pcd.paint_uniform_color([x / 255 for x in [255, 121, 111]])
            container_pcd, _ = container_pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=37.0)
            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=(0., 0., 0.))

            vis.clear_geometries()
            vis.add_geometry(pcd)
            vis.add_geometry(operator_pcd)
            vis.add_geometry(container_pcd)
            vis.add_geometry(line_set)
            vis.add_geometry(coordinate)
            for i in range(1):
                vis.poll_events()
                vis.update_renderer()
    vis.run()
