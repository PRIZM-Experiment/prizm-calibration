import numpy as np
import healpy
import scipy
from pygsm.pygsm2016 import GlobalSkyModel2016


class GSMData:

    def __init__(self, instrument, channel, min_per_bin, nside=256):
        self.min_per_bin = min_per_bin
        self.instrument = instrument
        self.channel = channel
        self.nside = nside
        self.beam_dict = self.get_beam_dict()
        self.healpy_beam = self.get_healpy_beam()
        self.gsm_data = None

    def __call__(self):
        self.get_GSM_temps().align_GSMdata()
        self.save_GSM_data()
        return self.gsm_data

    def get_beam_dict(self):
       
        dir_parent='./Beams'
        if self.instrument == '100MHz':
            file_name='results_pattern_100mhz_total90.dat'
        
        if self.instrument == '70MHz':
            file_name='results_pattern_70mhz_total90.dat'

        # Initializes the dictionary which will hold the beam information.
        beam_dict = {}

        # Establishes the `file_path` which points to the beam simulation of interest.
        file_path = dir_parent + '/' + file_name

        # Stores the beam simulation data in the NumPy array `beam_sim_data`, and
        # ignores the header as a comment starting with '#'.
        beam_sim_data = np.loadtxt(file_path, delimiter=',', comments='#')

        # Reads the beam file header, cleans it from unwanted characters, and keeps
        # only the numerical entries - these correspond to the different frequencies
        # for which the beam has been simulated.
        beam_file = open(file_path, 'r')
        header = beam_file.readline()
        frequencies = header.strip('#\n, ').split(',')[2:]
        beam_file.close()

        # Converts the `frequencies` list to a NumPy array and converts its values
        # to MHz through a division by 1e6.
        frequencies = np.asarray(frequencies, dtype='float') / 1e6

        # Extracts the spherial coordinates `theta` and `phi` stored in
        # `beam_sim_data` and converts their units from degrees to radians.
        theta = np.unique(beam_sim_data[:, 0]) * np.pi / 180
        phi = np.unique(beam_sim_data[:, 1]) * np.pi / 180

        # Discards the coordinate information from `beam_sim_data` since this is
        # already stored in the meshgrid, as well as in `theta` and `phi`.
        beam_sim_data = beam_sim_data[:, 2:]

        # Stores spherical coordinates in `beam_dict`.
        beam_dict['theta'] = theta
        beam_dict['phi'] = phi

        # Stores the beam profile for each frequency in `beam_dict`.
        for index, entry in enumerate(frequencies):
            # Reshape the `beam_sim_data` so that its dimensions are compatible with
            # those of `theta` and `phi`. This way different slices of `beam_sim_data`
            # correspond to the beam for different frequencies.
            reshaped_beam_sim = np.reshape(beam_sim_data[:, index],
                                           [len(phi), len(theta)])

            # Stores the reshaped beam in `beam_dict` under the appropriate
            # frequency key.
            beam_dict[entry] = reshaped_beam_sim

        # Returns the beam information in a dictionary format.
        return beam_dict

    def get_healpy_beam(self, site_latitude=-46.88694):
        # Initializes the dictionary which will hold the HealPy version of the beam.
        healpy_beam_dict = {}

        # Extracts the frequencies for which beams are available in `beam_dict`.
        frequencies = [key for key in self.beam_dict.keys() if isinstance(key, float)]
        n_freq = len(frequencies)

        # Initializes a HealPy pixelization and associated spherical coordinates.
        healpy_npix = healpy.nside2npix(self.nside)
        healpy_theta, healpy_phi = healpy.pix2ang(self.nside,
                                                  np.arange(healpy_npix))

        # Stores spherical coordinates in `healpy_beam_dict`.
        healpy_beam_dict['theta'] = healpy_theta
        healpy_beam_dict['phi'] = healpy_phi

        # SciPy 2D interpolation forces us to do proceed in chunks of constant
        # coordinate `healpy_theta`. Below we find the indices at which
        # `healpy_theta` changes.
        indices = np.where(np.diff(healpy_theta) != 0)[0]
        indices = np.append(0, indices + 1)

        # Initializes the NumPy array which will contain the normalization factor
        # for each beam.
        beam_norms = np.zeros(n_freq)

        # Loops over the different frequencies for which the beam has been
        # simulated.
        for i, frequency in enumerate(frequencies):

            # Computes the actual beam from the information contained in
            # `beam_dict`.
            beam = 10 ** (self.beam_dict[frequency] / 10)

            # Interpolates beam.
            beam_interp = scipy.interpolate.interp2d(self.beam_dict['theta'],
                                                     self.beam_dict['phi'],
                                                     beam,
                                                     kind='cubic',
                                                     fill_value=0)

            # Initializes `healpy_beam`, the HealPy version of the beam.
            healpy_beam = np.zeros(len(healpy_theta))

            # Constructs the HealPy beam.
            for j in range(int(len(indices) / 2) + 2):
                start = indices[j]
                end = indices[j + 1]
                healpy_beam[start:end] = beam_interp(healpy_theta[start],
                                                     healpy_phi[start:end])[:, 0]

            # Fills `beam_norms` with the appropriate normalization factors for
            # each HealPy beam.
            beam_norms[i] = np.sqrt(np.sum(healpy_beam ** 2))

            # Rotates and stores the the HealPy beam in the `healpy_beam_dict` under
            # the appropriate frequency entry.
            if self.channel == 'NS':
                beam_rotation = healpy.rotator.Rotator([0, 0, 90 - site_latitude])
            if self.channel == 'EW':
                beam_rotation = healpy.rotator.Rotator([90, 0, 90 - site_latitude])
            healpy_beam = beam_rotation.rotate_map_pixel(healpy_beam / beam_norms[i])
            healpy_beam_dict[frequency] = healpy_beam

        # Adds the beam normalizations as a separate entry in `heapy_beam_dict`.
        healpy_beam_dict['normalization'] = beam_norms

        # Returns the HealPy version of the beam in a dictionary format.
        return healpy_beam_dict

    @staticmethod
    def change_coord(m, coord):
        """ Change coordinates of a HEALPIX map

        Parameters
        ----------
        m : map or array of maps
          map(s) to be rotated
        coord : sequence of two character
          First character is the coordinate system of m, second character
          is the coordinate system of the output map. As in HEALPIX, allowed
          coordinate systems are 'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)

        Example
        -------
        The following rotate m from galactic to equatorial coordinates.
        Notice that m can contain both temperature and polarization.
        """
        # Basic HEALPix parameters
        npix = m.shape[-1]
        nside = healpy.npix2nside(npix)
        ang = healpy.pix2ang(nside, np.arange(npix))

        # Select the coordinate transformation
        rot = healpy.Rotator(coord=reversed(coord))

        # Convert the coordinates
        new_ang = rot(*ang)
        new_pix = healpy.ang2pix(nside, *new_ang)

        return m[..., new_pix]

    def get_GSM_temps(self, saved_maps=True):
        temperatures1 = []
        for i in range(30, 202, 2):
            if saved_maps:
                gsm_map_lowres = np.load(f'./gsm_maps/gsm_{i}.npy')
            else:
                gsm_2016 = GlobalSkyModel2016(freq_unit='MHz')
                gsm_map = gsm_2016.generate(i)
                gsm_map_eq = self.change_coord(gsm_map, ['G', 'C'])
                gsm_map_lowres = healpy.ud_grade(gsm_map_eq, self.nside, order_in='RING', order_out='RING')

            alm_map_eq = healpy.map2alm(gsm_map_lowres)
            alm_BEAM = healpy.map2alm(self.healpy_beam[i])
            temp_map = np.full(gsm_map_lowres.size, 1)
            alm_temp_map = healpy.map2alm(temp_map)
            integral_beam0 = np.real(np.sum(alm_temp_map * alm_BEAM))
            lmax = int(np.round(np.sqrt(2 * len(alm_BEAM) - 0.5)))
            m = np.zeros(len(alm_BEAM))

            icur = 0
            for i in range(0, lmax):
                nn = lmax - i
                m[icur:icur + nn] = i
                icur = icur + nn

            phi_rot1 = np.linspace(0, 2 * np.pi, int((1440 / self.min_per_bin) + 1))
            phitmp = phi_rot1.tolist()
            phitmp.pop()
            phi_rot1 = np.array(phitmp)

            temperatures0 = []
            for phi in phi_rot1:
                new_alm_beam = alm_BEAM * np.exp(-1j * phi * m)

                y0 = new_alm_beam * np.conj(alm_map_eq)

                integral_beam_map0 = np.real((np.sum(y0[:lmax]) + 2 * np.sum(y0[lmax:])))

                amp_alm_space0 = integral_beam_map0 / integral_beam0

                temperatures0.append(amp_alm_space0)

            temperatures1.append(temperatures0)
        self.gsm_data = np.array(temperatures1).T
        return self

    def align_GSMdata(self):
        bin1 = round(1440 / self.min_per_bin)
        I1 = int((360 / 720) * bin1)
        I2 = int((530 / 720) * bin1)
        TGSM = []
        zerobin = (bin1 - I2) + I1
        for i in range(zerobin, bin1):
            T2 = self.gsm_data[i, :].tolist()
            TGSM.append(T2)
        for i in range(0, zerobin):
            T2 = self.gsm_data[i, :].tolist()
            TGSM.append(T2)
        self.gsm_data = np.array(TGSM)
        return self
    
    def save_GSM_data(self):
        np.save(f'./GSM_averages/{self.instrument}_{self.channel}_GSM_average_{self.min_per_bin}min', self.gsm_data)

    def save_GSM_maps(self):
        gsm_2016 = GlobalSkyModel2016(freq_unit='MHz')

        for i in range(30, 202, 2):
            gsm_map = gsm_2016.generate(i)
            gsm_map_eq = self.change_coord(gsm_map, ['G', 'C'])
            gsm_map_lowres = healpy.ud_grade(gsm_map_eq, 256, order_in='RING', order_out='RING')
            np.save(f'gsm_{i}', gsm_map_lowres)

            
def get_desired_frequencies(Tgsm, flow, fhigh):
    start = int((flow-30)/2)
    end = int((fhigh-flow)/2) + start +1
    return Tgsm[:, start:end]
    