import open3d as o3d
import argparse
import os

def ply_to_mesh(input_ply, output_ply, depth=9):
    """
    将 Cap4D 导出的点云 ply 转换为三角网格 ply
    使用 Poisson Surface Reconstruction 方法
    """
    if not os.path.exists(input_ply):
        raise FileNotFoundError(f"输入文件不存在: {input_ply}")

    # 读取点云
    print(f"读取点云: {input_ply}")
    pcd = o3d.io.read_point_cloud(input_ply)

    # 法向量估计（Poisson 需要）
    print("估计法向量...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    # Poisson surface reconstruction
    print("进行 Poisson 网格重建...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    
    # 可选：去除低密度三角形
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # 保存网格
    print(f"保存网格到: {output_ply}")
    o3d.io.write_triangle_mesh(output_ply, mesh)
    print("完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 Cap4D 点云 ply 转换为网格 ply")
    parser.add_argument("--input", type=str, required=True, help="输入 ply 点云路径")
    parser.add_argument("--output", type=str, required=True, help="输出 ply 网格路径")
    parser.add_argument("--depth", type=int, default=9, help="Poisson 重建深度，越大网格越精细（内存占用增大）")
    args = parser.parse_args()

    import numpy as np
    ply_to_mesh(args.input, args.output, args.depth)

# python tools/ply_to_mesh.py \
#     --input examples/output/tesla/animation_00/exported_animation.ply \
#     --output examples/output/tesla/animation_00/exported_animation_mesh.ply \
#     --depth 10