import numpy as np
import pandas as pd
from astropy.wcs import WCS

import psycopg2
import psycopg2.extras
import re

proposals_ = ["2020B-0053", "2021A-0275", "2022A-388025", "2021A-0113", "2021B-0149", "2022A-724693"]


class Candidate:

    def __init__(self, candidate, radec,
                 proposals=None):
        if proposals is None:
            proposals = proposals_
        self.name = candidate
        self.ra = radec[0]
        self.dec = radec[1]
        self.fp_base = "/global/cfs/cdirs/m937/www/decat/"
        self.imdfs = None
        self.cursor = None
        self.q = ""
        self.proposals = proposals
        self.c = None

    def start(self):
        """
        set the query to get all images and start the cursor.
        """
        
        # username = 
        # password = 

        db = psycopg2.connect(dbname="decat", host="decatdb.lbl.gov", port=5432, user=username, password=password,
                              cursor_factory=psycopg2.extras.RealDictCursor)
        self.q = (
                    "SELECT i.id,i.basename,e.proposalid,e.filter FROM images i INNER JOIN exposures e ON i.exposure_id=e.id "
                    "WHERE i.ra1<" + str(self.ra) + " AND i.ra2>" + str(self.ra) + " AND i.dec1>" + str(
                self.dec) + " AND i.dec<" + str(self.dec) + " ORDER BY basename")
        self.cursor = db.cursor()

    def close(self):
        """
        close the cursor
        """
        self.cursor.close()

    def get_image_ids(self):
        """
        execute the query to get all images and get the database which contains the file
        basename of all the images.
        """
        self.cursor.execute(self.q)
        rows = self.cursor.fetchall()
        self.imdfs = pd.DataFrame(rows)
        self.imdfs = self.imdfs[self.imdfs["proposalid"].isin(self.proposals)]

        return self.imdfs

    def get_fp(self, imid):
        """
        Get full the filepath given an image id in the database.
        Catches error if filepath doesn't exist at that proposal.
        """
        try:
            fpdf = self.imdfs[self.imdfs['basename'].str.contains(imid)]

            basename = fpdf.basename.iloc[0]
            propid = fpdf.proposalid.iloc[0]
            date = re.search('^c4d_(\d{6})_(\d{6})_ori.(\d{2}).fits$', basename).group(1)

            imfp = self.fp_base + propid + "/" + date + "/" + basename[:-8] + "/" + basename + ".fz"
            wfp = self.fp_base + propid + "/" + date + "/" + basename[:-8] + "/" + basename[:-4] + "weight.fits.fz"

            return imfp, wfp

        except IndexError:
            print("Image ID not in index " + imid)

    def get_all_fps(self):
        """
        Gets full filepaths for all of the image ids in the database.
        """
        fps = []
        for b in self.imdfs.basename:
            fps.append(self.get_fp(b))

        return fps


class Quadrant:

    def __init__(self, hdul, quadrant_size):
        self.im = hdul[1].data
        self.hdul = hdul
        self.imsize = self.im.shape
        self.qsize = quadrant_size
        self.band = hdul[1].header["FILTER"][0]
        self.width = int(self.im.shape[0] / self.qsize[0])
        self.height = int(self.im.shape[1] / self.qsize[1])
        self.center = (0, 0)
        self.left_corner = (0, 0)

    def get_band(self):
        """
        Get the band where from the image.
        """
        return self.band

    def get_radec_from_xy(self, pos):
        """
        given an numpy array of Ra/Decs, return a numpy array of positions on the full image.
        The positions are using a numpy array, but because you use them as X and Y coordinates,
        the first number (a in (a, b)) is the X position on the graph, and the second is the y position.
        The origin is (0, 0) located at the top left corner, and both axes to the right and down are positive.

        So, you use it as an (x, y) coordinate in plt.scatter()
        """

        w = WCS(self.hdul[1].header)  # create a wcs object with the quadrant object's header
        return w.wcs_pix2world(pos, 0)  # return the pixel positions as a numpy array

    def get_xy_from_radec(self, pos):
        """
        given an numpy array of X/Y pixel positions, return a numpy array of positions on the full image.
        The positions are using a numpy array, but because you use them as X and Y coordinates,
        the first number (a in (a, b)) is the X position on the graph, and the second is the y position.
        The origin is (0, 0) located at the top left corner, and both axes to the right and down are positive.

        So, you use the Xs and Ys as an (x, y) coordinate in plt.scatter()
        """

        w = WCS(self.hdul[1].header)  # create a wcs object with the quadrant object's header
        return w.wcs_world2pix(pos, 0)  # return the Ra/Dec positions as a numpy array

    def get_centered_im(self, ra, dec):
        """
        Get an image centered around the integer nearest to the ra/dec transformed into pixels with
        wcs.
        """
        xy = self.get_xy_from_radec(np.array([[ra, dec]]))  # Get the pixel value for the Ra/Dec with astropy

        xy = xy[0]  # Get the first entry in the numpy array of pixel coordinates (the array has just one coordinate)
        x = int(xy[0])  # Get the center positions of the tile. This coordinate is (x, y) with plt.scatter()
        y = int(xy[1])

        self.center = (x, y)
        ylength = int(self.qsize[1] / 2)  # The length of the quadrant on the downwards axis
        xlength = int(self.qsize[0] / 2)  # The length of the quadrant on the horizontal axis

        # The quadrant is centered around int(x) and int(y) and will either make a square or hit the edge of the image.

        ymin = y - min(y, ylength)  # The minimum bound on the downwards axis
        ymax = y + min(self.im.shape[0] - y, ylength)  # The maximum bound on the downwards axis

        # Note to self: Numpy arrays index with [rows, columns], so shape is (#rows (length of y axis), #columns (length of x axis))

        xmin = x - min(x, xlength)  # The minimum bound on the horizontal axis
        xmax = x + min(self.im.shape[1] - x, xlength)  # The maximum bound on the horizontal axis

        self.left_corner = (xmin, ymin)

        tile = self.im[ymin:ymax, xmin:xmax]  # Index the rows (downwards axis), and then the columns (horizontal axis)

        return tile, ymin, ymax, xmin, xmax
        # When using the bounds as POSITIONS, do (xmin, ymin) to get POSITIONS. When using them to index the IMAGE, do (ymin, xmin) --> (rows, columns).

    def get_rel_pos(self, x, y):
        """
        After defining a centered image, get the relative positions of xs and ys from the full (4096, 2048) sized image
        to the quadrant.
        """

        xrel = x - self.left_corner[0]
        yrel = y - self.left_corner[1]

        return xrel, yrel
    
    
    def get_tile(self, radec):
        size = int(self.qsize[0]/2)
        xy = self.get_xy_from_radec(np.array([[radec[0], radec[1]]])) # convert radec to pixel values
        c = (int(xy[0][0]-size), int(xy[0][0]+size), int(xy[0][1]-size), int(xy[0][1]+size))

        subim = self.im
        return np.flipud(subim[c[2]:c[3], c[0]:c[1]]), c
    
    def get_rel_xy(self, radec):
        """
        JUST FOR FLOAT RA DECS
        """
        
        size = int(self.qsize[0]/2) # Right now only handles square tiles...?
        array = [[radec[0], radec[1]]]
        xy = np.transpose(self.get_xy_from_radec(array)) # convert radec to pixel values

        c = (int(xy[0]-size), int(xy[0]+size), int(xy[1]-size), int(xy[1]+size))
        self.c = c

        xrel = xy[0] - c[0]
        yrel = xy[1] - c[2]
        yrel = 2*size - yrel-1

        return xrel, yrel
    
    def get_rel_xys(self, x, y):
        c = self.c
        
        xrel = x - c[0]
        yrel = y - c[2]
        
        return xrel, yrel