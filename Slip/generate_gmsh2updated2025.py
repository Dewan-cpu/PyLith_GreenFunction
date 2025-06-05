#!/usr/bin/env nemesis
"""Generate a tri of strike-slip faults using Gmsh."""

import numpy
import gmsh
from pylith.meshio.gmsh_utils import (VertexGroup, MaterialGroup, GenerateMesh)

class App(GenerateMesh):
    DOMAIN_W = 147771.00  # previously 247771.00
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
        super().__init__()
        self.cell_choices = {"default": "tri", "choices": ["tri", "quad"]}
        self.filename = "mesh_tri2.msh"

    def _create_points_from_file(self, filename):
        coordinates = numpy.loadtxt(filename)
        return [gmsh.model.geo.add_point(x, y, 0) for x, y in coordinates]

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

        self._add_faults()

        gmsh.model.geo.remove_all_duplicates()
        gmsh.model.geo.synchronize()

    def _add_faults(self):
        def add_fault(file, name, idx_intersect=None):
            points = self._create_points_from_file(file)
            if idx_intersect is not None:
                spline = gmsh.model.geo.add_spline(points)
                return gmsh.model.geo.split_curve(spline, [points[idx_intersect]])
            return [gmsh.model.geo.add_spline(points)]

        # Add main faults
        self.curves_fault_main_1 = add_fault(self.FILENAME_MAINTRACE, "main_1", 3)
        self.curves_fault_main_2 = add_fault(self.FILENAME_MAINTRACE2, "main_2")

        # Add east and south faults
        self.c_fault_east_1 = gmsh.model.geo.add_spline(self._create_points_from_file(self.FILENAME_EAST1TRACE))
        self.c_fault_east_2 = gmsh.model.geo.add_spline(self._create_points_from_file(self.FILENAME_EAST2TRACE))
        self.c_fault_east_3 = gmsh.model.geo.add_spline(self._create_points_from_file(self.FILENAME_EAST3TRACE))
        self.c_fault_south = gmsh.model.geo.add_spline(self._create_points_from_file(self.FILENAME_SOUTHTRACE))

        # Manual split of curves by ID (if known)
        gmsh.model.geo.split_curve(7, [20, 17, 7])
        gmsh.model.geo.split_curve(8, [29])

    def mark(self):
        materials = (MaterialGroup(tag=1, entities=[self.s_domain]),)
        for material in materials:
            material.create_physical_group()

        vertex_groups = (
            VertexGroup(name="boundary_south", tag=10, dim=1, entities=[self.c_south]),
            VertexGroup(name="boundary_east", tag=11, dim=1, entities=[self.c_east]),
            VertexGroup(name="boundary_north", tag=12, dim=1, entities=[self.c_north]),
            VertexGroup(name="boundary_west", tag=13, dim=1, entities=[self.c_west]),
            VertexGroup(name="fault_main_1", tag=20, dim=1, entities=self.curves_fault_main_1),
            VertexGroup(name="fault_main_2", tag=24, dim=1, entities=self.curves_fault_main_2),
            VertexGroup(name="fault_east_1", tag=21, dim=1, entities=[self.c_fault_east_1]),
            VertexGroup(name="fault_east_2", tag=22, dim=1, entities=[self.c_fault_east_2]),
            VertexGroup(name="fault_east_3", tag=23, dim=1, entities=[self.c_fault_east_3]),
            VertexGroup(name="fault_south", tag=34, dim=1, entities=[self.c_fault_south]),
        )
        for group in vertex_groups:
            group.create_physical_group()

    def generate_mesh(self, cell):
        gmsh.option.set_number("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.set_number("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)

        field_distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            field_distance, "CurvesList",
            self.curves_fault_main_1 +
            [self.c_fault_east_1, self.c_fault_east_2, self.c_fault_east_3] +
            self.curves_fault_main_2 + [self.c_fault_south]
        )

        field_size = gmsh.model.mesh.field.add("MathEval")
        math_exp = GenerateMesh.get_math_progression(field_distance, min_dx=self.DX_FAULT, bias=self.DX_BIAS)
        gmsh.model.mesh.field.setString(field_size, "F", math_exp)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_size)
        gmsh.model.mesh.generate(2)

if __name__ == "__main__":
    App().main()
