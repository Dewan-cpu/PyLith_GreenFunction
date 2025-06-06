#!/usr/bin/env nemesis
"""This is an extremely simple example showing how to set up an inversion
using PyLith-generated Green's functions. In this simple example,
there are no data uncertainties, and we use the minimum moment for a penalty function.
"""

import argparse
import numpy
import h5py

class InvertSlipApp:
    """Application to invert for fault slip using PyLith-generated Green's functions.
    """
    def __init__(self):
        self.filename_fault = "output/step05_greensfns-fault_main.h5"
        self.filename_responses = "output/step05_greensfns-gnss_stations.h5"
        self.filename_observed = "output/gnss_stations2.txt"
        self.filename_output = "output/step06_inversion-results.txt"
        self.penalties = [0.01, 0.1, 1.0]

    def main(self):
        args = self._parse_command_line()

        self.get_fault_impulses(args.filename_fault)
        self.get_station_responses(args.filename_responses)
        self.get_station_observed(args.filename_observed)

        self.penalties = list(map(float, args.penalties.split(",")))
        results = self.invert()
        self.write_results(self.filename_output, results)

    def get_fault_impulses(self, filename):
        h5 = h5py.File(filename, "r")
        y = h5['geometry/vertices'][:,1]
        slip = h5['vertex_fields/slip'][:,:,1]
        h5.close()

        reorder = numpy.argsort(y)
        self.impulse_y = y[reorder]
        self.impulse_slip = slip[:,reorder]

    def get_station_responses(self, filename):
        h5 = h5py.File(filename, "r")
        xy = h5['geometry/vertices'][:,0:2]
        displacement = h5['vertex_fields/displacement'][:]  # shape (nimpulses, nstations, ncomponents)
        h5.close()

        reorder = numpy.argsort(xy[:,1])
        self.station_responses = displacement[:,reorder,:]

    def get_station_observed(self, filename):
        data = numpy.loadtxt(filename, encoding="utf-8", usecols=(1,2,3,4,5))
        # Only use East (col 3) and North (col 2) components
        displacement = numpy.vstack((data[:,3], data[:,2])).T
        xy = data[:, 0:2]

        reorder = numpy.argsort(xy[:,1])
        self.station_observed = displacement[reorder,:]

    def invert(self):
        nfaultpts = self.impulse_y.shape[0]
        nimpulses = self.station_responses.shape[0]
        nobs = self.station_responses.shape[1] * self.station_responses.shape[2]
        mat_A = self.station_responses.reshape((nimpulses, nobs)).T

        vec_data = numpy.concatenate((
            self.station_observed.flatten(),
            numpy.zeros(nimpulses, dtype=numpy.float64)
        ))

        ninversions = len(self.penalties)
        results = numpy.zeros((nfaultpts, 1 + ninversions))
        results[:,0] = self.impulse_y

        for i, penalty in enumerate(self.penalties):
            mat_penalty = penalty * numpy.eye(nimpulses, dtype=numpy.float64)
            mat_design = numpy.vstack((mat_A, mat_penalty))

            mat_gen_inverse = numpy.dot(
                numpy.linalg.inv(numpy.dot(mat_design.T, mat_design)),
                mat_design.T
            )

            impulse_amplitude = numpy.dot(mat_gen_inverse, vec_data)
            slip_soln = numpy.dot(impulse_amplitude, self.impulse_slip)
            results[:, 1+i] = slip_soln

            predicted = numpy.dot(mat_A, impulse_amplitude)
            residual = self.station_observed.flatten() - predicted
            residual_norm = numpy.linalg.norm(residual)
            print(f"Penalty parameter:  {penalty}")
            print(f"Residual norm:      {residual_norm}")

        return results

    def write_results(self, filename, results):
        header = "# y"
        for penalty in self.penalties:
            header += f" penalty={penalty}"
        header += "\n"

        with open(filename, "w") as fout:
            fout.write(header)
            numpy.savetxt(fout, results, fmt="%14.6e")

    def _parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--impulse-data", action="store", type=str,
                            dest="filename_fault", default=self.filename_fault,
                            help="Fault HDF5 data file from Green's function simulation.")

        parser.add_argument("--impulse-responses", action="store", type=str,
                            dest="filename_responses", default=self.filename_responses,
                            help="Station HDF5 data file from Green's function simulation.")

        parser.add_argument("--observed-data", action="store", type=str,
                            dest="filename_observed", default=self.filename_observed,
                            help="Station TXT data file from variable slip simulation.")

        parser.add_argument("--penalties", action="store", type=str,
                            dest="penalties", default="0.01,0.1,1.0",
                            help="Comma-separated list of penalties.")

        parser.add_argument("--output", action="store", type=str,
                            dest="filename_output", default=self.filename_output,
                            help="Name of output file with inversion results.")

        return parser.parse_args()


if __name__ == "__main__":
    InvertSlipApp().main()

# End of file
