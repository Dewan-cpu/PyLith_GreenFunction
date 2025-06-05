#!/usr/bin/env nemesis
"""Generate a tri of strike-slip faults using Gmsh."""

import numpy
import gmsh
from pylith.meshio.gmsh_utils import (VertexGroup, MaterialGroup, GenerateMesh)

class App(GenerateMesh):
    DOMAIN_W = 157771.00  # previously 247771.00
    DOMAIN_E = 505833.00
    DOMAIN_S = 3994700.00
    DOMAIN_N = 4267080.00

    FILENAME_MAINTRACE = "faulttrace_main_utm.txt"
    FILENAME_EAST1TRACE = "faulttrace_east1_utm.txt"
    FILENAME_EAST2TRACE = "faulttrace_east2_utm.txt"
    FILENAME_EAST3TRACE = "faulttrace_east3_utm.txt"
    FILENAME_MAINTRACE2 = "faulttrace_main_2_utm.txt"
    FILENAME_SOUTHTRACE = "faulttrace_south_utm.txt"

    DX_FAULT = 1.0e+3
    DX_BIAS = 1.07

    def __init__(self):
        """Constructor."""
        super().__init__()

        self.cell_choices = {
            "default": "tri",
            "choices": ["tri", "quad"],
        }
        self.filename = "mesh_tri.msh"

    def _create_points_from_file(self, filename):
        coordinates = numpy.loadtxt(filename)
        points = []
        for xy in coordinates:
            points.append(gmsh.model.geo.add_point(xy[0], xy[1], 0))
        return points

    def create_geometry(self):
        pSW = gmsh.model.geo.add_point(self.DOMAIN_W, self.DOMAIN_S, 0)
        pSE = gmsh.model.geo.add_point(self.DOMAIN_E, self.DOMAIN_S, 0)
        pNE = gmsh.model.geo.add_point(self.DOMAIN_E, self.DOMAIN_N, 0)
        pNW = gmsh.model.geo.add_point(self.DOMAIN_W, self.DOMAIN_N, 0)

        self.c_south = gmsh.model.geo.add_line(pSW, pSE)
        self.c_east = gmsh.model.geo.add_line(pSE, pNE)
        self.c_north = gmsh.model.geo.add_line(pNE, pNW)
        self.c_west = gmsh.model.geo.add_line(pNW, pSW)

        loop = gmsh.model.geo.add_curve_loop([self.c_south, self.c_east, self.c_north, self.c_west])
        self.s_domain = gmsh.model.geo.add_plane_surface([loop])

        points = self._create_points_from_file(self.FILENAME_MAINTRACE)
        p_fault_intersection = [points[2], points[12], points[15]]
        spline = gmsh.model.geo.add_spline(points)
        self.curves_fault_main = gmsh.model.geo.split_curve(spline, p_fault_intersection)
        self.fault_main_ends = [points[0], points[-1]]

        points = self._create_points_from_file(self.FILENAME_EAST1TRACE)
        self.c_fault_east1 = gmsh.model.geo.add_spline(points)

        points = self._create_points_from_file(self.FILENAME_EAST2TRACE)
        self.c_fault_east2 = gmsh.model.geo.add_spline(points)

        points = self._create_points_from_file(self.FILENAME_EAST3TRACE)
        self.c_fault_east3 = gmsh.model.geo.add_spline(points)

        points = self._create_points_from_file(self.FILENAME_MAINTRACE2)
        p_fault_intersection = [points[5]]
        spline = gmsh.model.geo.add_spline(points)
        self.curves_fault_main2 = gmsh.model.geo.split_curve(spline, p_fault_intersection)
        self.fault_main2_ends = [points[0], points[-1]]

        points = self._create_points_from_file(self.FILENAME_SOUTHTRACE)
        self.c_fault_south = gmsh.model.geo.add_spline(points)

        gmsh.model.geo.remove_all_duplicates()
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(1, self.curves_fault_main.tolist(), 2, self.s_domain)
        gmsh.model.mesh.embed(1, [self.c_fault_east1], 2, self.s_domain)
        gmsh.model.mesh.embed(1, [self.c_fault_east2], 2, self.s_domain)
        gmsh.model.mesh.embed(1, [self.c_fault_east3], 2, self.s_domain)
        gmsh.model.mesh.embed(1, self.curves_fault_main2.tolist(), 2, self.s_domain)
        gmsh.model.mesh.embed(1, [self.c_fault_south], 2, self.s_domain)
        gmsh.model.geo.synchronize()

        _, self.fault_east1_ends = gmsh.model.get_adjacencies(dim=1, tag=self.c_fault_east1)
        _, self.fault_east2_ends = gmsh.model.get_adjacencies(dim=1, tag=self.c_fault_east2)
        _, self.fault_east3_ends = gmsh.model.get_adjacencies(dim=1, tag=self.c_fault_east3)
        _, self.fault_south_ends = gmsh.model.get_adjacencies(dim=1, tag=self.c_fault_south)

    def mark(self):
        """Mark geometry for materials, boundary conditions, faults, etc."""
        materials = (
            MaterialGroup(tag=1, entities=[self.s_domain]),
        )
        for material in materials:
            material.create_physical_group()

        vertex_groups = (
            VertexGroup(name="boundary_south", tag=10, dim=1, entities=[self.c_south]),
            VertexGroup(name="boundary_east", tag=11, dim=1, entities=[self.c_east]),
            VertexGroup(name="boundary_north", tag=12, dim=1, entities=[self.c_north]),
            VertexGroup(name="boundary_west", tag=13, dim=1, entities=[self.c_west]),

            VertexGroup(name="fault_main", tag=20, dim=1, entities=self.curves_fault_main.tolist()),
            VertexGroup(name="fault_main_2", tag=21, dim=1, entities=self.curves_fault_main2.tolist()),
            VertexGroup(name="fault_east_1", tag=22, dim=1, entities=[self.c_fault_east1]),
            VertexGroup(name="fault_east_2", tag=23, dim=1, entities=[self.c_fault_east2]),
            VertexGroup(name="fault_east_3", tag=24, dim=1, entities=[self.c_fault_east3]),
            VertexGroup(name="fault_south", tag=25, dim=1, entities=[self.c_fault_south]),
            VertexGroup(name="fault_main_ends", tag=30, dim=0, entities=self.fault_main_ends),
            VertexGroup(name="fault_main2_ends", tag=31, dim=0, entities=self.fault_main2_ends),
            VertexGroup(name="fault_east1_ends", tag=32, dim=0, entities=self.fault_east1_ends),
            VertexGroup(name="fault_east2_ends", tag=33, dim=0, entities=self.fault_east2_ends),
            VertexGroup(name="fault_east3_ends", tag=34, dim=0, entities=self.fault_east3_ends),
            VertexGroup(name="fault_south_ends", tag=35, dim=0, entities=self.fault_south_ends),
        )
        for group in vertex_groups:
            group.create_physical_group()

    def generate_mesh(self, cell):
        """Generate the mesh."""
        gmsh.option.set_number("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.set_number("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)

        field_distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            field_distance, "CurvesList",
            list(self.curves_fault_main) + [
                self.c_fault_east1,
                self.c_fault_east2,
                self.c_fault_east3,
                *list(self.curves_fault_main2),
                self.c_fault_south
            ]
        )

        field_size = gmsh.model.mesh.field.add("MathEval")
        math_exp = GenerateMesh.get_math_progression(
            field_distance, min_dx=self.DX_FAULT, bias=self.DX_BIAS
        )
        gmsh.model.mesh.field.setString(field_size, "F", math_exp)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_size)

        if cell == "quad":
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.recombine()
        else:
            gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Laplace2D")

if __name__ == "__main__":
    App().main()

# End of file
