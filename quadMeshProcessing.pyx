# distutils: language = c++
import cython
from scipy.spatial import kdtree, Rectangle
import numpy as np
cimport numpy as np

from cython.parallel import prange
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI
from libc.stdio cimport printf

np.import_array()

cdef class quadMeshProcessing :

    cdef int verbose
    cdef double[:,:] verts_mv
    cdef double[:,:,:] quads_mv 
    cdef int[:,:] faces_mv
    cdef int[:] verts_label_map_mv 
    cdef int[:] faces_label_map_mv
    cdef double[:,:] normals_mv
    cdef vector[vector[vector[double]]] vquads
    cdef readonly object normals
    cdef readonly object quads
    cdef readonly object vertices
    cdef readonly object faces
    cdef readonly object vertices_label_map
    cdef readonly object faces_label_map
    cdef readonly object face_connectivity_matrix

    def __init__(self, verts, faces, 
                    verts_label_map=None, faces_label_map=None, verbose=0):
        self.verbose = verbose
        self.faces = faces.copy()
        self.vertices = verts.copy()
        self.vertices_label_map = verts_label_map.copy() if verts_label_map is not None \
                                    else np.arange(self.verts.shape[0], dtype="int32")
        self.faces_label_map = faces_label_map.copy() if faces_label_map is not None \
                                    else np.arange(self.faces.shape[0], dtype="int32")
        self.normals = np.zeros((self.faces.shape[0], 3), dtype="float64")

        self.setMemoryViews(self.faces, self.vertices, 
                            self.vertices_label_map, self.faces_label_map,
                            self.normals)
        self.face_connectivity_matrix = np.zeros((self.faces.shape[0], 
                                                  self.faces.shape[0]), dtype='uint8')

        self.quads = self.getQuads()
        self.correct_winding_faces()
        self.computeNormals()

    cdef void setMemoryViews(self, faces, verts, 
                    verts_label_map, faces_label_map, normals):
        self.verts_mv = verts.astype(dtype="float64", subok=True, copy=False)
        self.faces_mv = faces.astype(dtype="int32", subok=True, copy=False)
        self.verts_label_map_mv = verts_label_map.astype(dtype="int32", subok=True, copy=False)
        self.faces_label_map_mv =  faces_label_map.astype(dtype="int32", subok=True, copy=False)       
        self.normals_mv =  normals.astype(dtype="float64", subok=True, copy=False)     


    cdef fill_connectivity_matrix(self):
        """Here we have to check, for each face, which faces share the same nodes
        """
        pass


    def getGlobalBounds(self):
        """Returns the bounds of the mesh in its absolute coordinates
        """
        if self.normals_mv.shape[0] != self.faces_mv.shape[0] :
            if self.quads_mv.shape[0] != self.faces_mv.shape[0] :
                self.quads = self.getQuads()
                self.computeNormals()
            else : 
                self.computeNormals()
        x_min = np.min(self.quads_mv[:,:,0])
        x_max = np.max(self.quads_mv[:,:,0])
        y_min = np.min(self.quads_mv[:,:,1])
        y_max = np.max(self.quads_mv[:,:,1])
        z_min = np.min(self.quads_mv[:,:,2])
        z_max = np.max(self.quads_mv[:,:,2])
        bounds = np.asarray([[x_min ,x_max], [y_min ,y_max], [z_min ,z_max]])
        return bounds

    def getLocalBounds(self):
        """Returns the bounds, but in the relative coordinate system. The minimal
        values are all 0 
        """
        bounds = self.getGlobalBounds()
        relative_bounds = np.subtract(bounds, np.expand_dims(bounds[:, 0], axis = 1))   
        return relative_bounds

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void computeNormals(self) nogil:
        """We have to correct the windings, so that each pair of consecutive nodes in the faces
        represent a edge (if the perimeter is abcd, we should no have acbd for example)
        """
        cdef size_t i, j, n_quads
        cdef vector[double] normal_vect

        n_quads = self.quads_mv.shape[0]

        for i in prange(n_quads):
            normal_vect.clear()
            normal_vect = quadNormal(self.quads_mv[i])
            for j in range(3):
                self.normals_mv[i,j]=normal_vect[j]
        if self.verbose > 0 :
            printf('    Normals computed')


    cdef correct_winding_faces(self):
        """We have to correct the windings, so that each pair of consecutive nodes in the faces
        represent a edge (if the perimeter is abcd, we should no have acbd for example)
        """
        cdef size_t i, n_quads
        cdef Py_ssize_t[4] basic_indexing = [0,1,2,3]
        cdef Py_ssize_t[:] new_indexing 
        cdef double[:,:] quad_mv
        cdef bool isSame
        cdef int isCooplanar
        cdef SymmetricMatrix quad_distance_matrix = SymmetricMatrix(4)

        n_quads = self.quads_mv.shape[0]

        for i in range(n_quads):
            quad_mv = self.quads_mv[i,:,:]
            new_indexing = self.correct_winding_1face(quad_mv, quad_distance_matrix)
            isCooplanar = self.checkIfFaceCooplanar(quad_mv)
            #if isCooplanar == 0 :
            #    print('Is not cooplanar')
            isSame = IntListComparison(new_indexing, basic_indexing, 4)
            if isSame == False :
                print('\tWe ll switch index 1 and 2...')
                temp0 = self.faces[i,1]
                temp1 = self.faces[i,2]
                self.faces[i,1] = temp1 
                self.faces[i,2] = temp0 #inverting
        self.setMemoryViews(self.faces, self.vertices, 
                self.vertices_label_map, self.faces_label_map, self.normals)
        self.quads = self.getQuads()
        if self.verbose > 0 :
            print('    winding corrected')


    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef int checkIfFaceCooplanar(self, double[:,:] quad, double thresh = 1e-10) nogil:
        cdef double a,b,c,d, a1, b1, c1, a2, b2, c2, dist
        cdef bool isCooplanar
        a1 = quad[1,0] - quad[0,0]
        b1 = quad[1,1] - quad[0,1]
        c1 = quad[1,2] - quad[0,2]
        a2 = quad[2,0] - quad[0,0]
        b2 = quad[2,1] - quad[0,1]
        c2 = quad[2,2] - quad[0,2]
        a = b1 * c2 - b2 * c1
        b = a2 * c1 - a1 * c2
        c = a1 * b2 - b1 * a2
        d = (- a * quad[0,0] - b * quad[0,1] - c * quad[0,2])

        dist = fabs(a * quad[3,0] + b * quad[3,1] + c * quad[3,2] + d)/sqrt(a**2+b**2+c**2)
        isCooplanar = (a * quad[3,0] + b * quad[3,1] + c * quad[3,2] + d == 0)
        if  dist < thresh:
            return 1 
        else: 
            return 0

    cdef Py_ssize_t[:] correct_winding_1face(self, double[:,:] quad_face, SymmetricMatrix distance_matrix ):
        """Checks the winding for one face and returns the new
        order of the indices 
        """
        cdef :
            Py_ssize_t[4] basic_indexing = [0,1,2,3]
            Py_ssize_t[4] new_indexing = [0,1,2,3]
            Py_ssize_t[8][2] edgeIdx = [[0,1], [1,2], [2,3], [3,0], [1,0], [2,1], [3,2], [0,3]]
            Py_ssize_t[4][2] diagIdx = [[0,2], [1,3], [2,0], [3,1]]
            Py_ssize_t[:] e
            Py_ssize_t[2] arg_max, edge
            Py_ssize_t[:] arg_max_mv
            Py_ssize_t[:,:] edgeIdx_mv, diagIdx_mv
            Py_ssize_t inEdgeIdx, temp0, temp1, inDiagIdx
            size_t len_l, len_ll, len_diagIdx, i, i_diag, i_diag_idx
            vector[bool] diff 
            bool sameVal0, sameVal1

        intra_face_distance_matrix(quad_face, distance_matrix)
        # we keep the first node at the right index, and then we check the distance betwwen the 
        # different nodes. We know that the max distance represents a diagonal. 
        # (at least one diagonal is the max) if the quad element is CONVEX
        arg_max_vec = distance_matrix.cargmax()
        arg_max[0] = arg_max_vec[0]
        arg_max[1] = arg_max_vec[1]
        edgeIdx_mv = edgeIdx
        diagIdx_mv = diagIdx
        arg_max_mv = arg_max

        len_l = 2 
        len_ll = 8
        len_diagIdx = 4
        # The maximal distance between two points should never be
        # between adjacent edges.
        inEdgeIdx = IntListInNestedIntList(arg_max_mv, edgeIdx_mv, len_l, len_ll )
        inDiagIdx = IntListInNestedIntList(arg_max_mv, diagIdx_mv, len_l, len_diagIdx )

        if inEdgeIdx >= 0:
            # This means  one of the adjacent edges is a diagonal, which of course is impossible.
            # We have to invert two indices, so that the winding gets correct
            # We have to replace one of the indices in arg_max, so that arg_max becomes part of 
            # diagIdx
            #Now we search the first transform that does the job.

            edge = edgeIdx[inEdgeIdx]

            for i in range(4):
                diff.clear()
                sameVal0 = edge[0]!=diagIdx[i][0]
                sameVal1 = edge[1]!=diagIdx[i][1] 
                diff.push_back(sameVal0)
                diff.push_back(sameVal1)
                if diff[0] != diff[1]:
                    i_diag = i
                    if diff[0] == True:
                        i_diag_idx = 0
                    else :
                        i_diag_idx = 1 
                diff.clear()
            
            #print('We have to switch index',edge[i_diag_idx],'with',diagIdx[i_diag][i_diag_idx])

            new_indexing[edge[i_diag_idx]] = basic_indexing[diagIdx[i_diag][i_diag_idx]]
            new_indexing[diagIdx[i_diag][i_diag_idx]] = basic_indexing[edge[i_diag_idx]]

            return new_indexing

        if inDiagIdx >= 0 :
            return basic_indexing

        #else :
        #    raise Exception("Shit happened")
        
    #########################################################################

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef getQuads(self):
        cdef size_t[:] fs = np.asarray(self.faces_mv.shape, dtype = np.uint)
        cdef np.ndarray[np.float64_t, ndim=3] nquads = np.zeros(shape=(fs[0], fs[1], 3), dtype = "float64")
        cdef size_t i, j, p
        self.vquads.clear()
        self.vquads = cgetQuads(self.faces_mv, fs, self.verts_mv, self.verts_label_map_mv)
        for i in range(fs[0]):
            for j in range(fs[1]):
                for p in range(3):
                    nquads[i,j,p] = self.vquads[i][j][p]
        self.vquads.clear()
        self.quads_mv = nquads.astype(dtype="float64", subok=True, copy=False)
        return nquads  

#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef Py_ssize_t IntListInNestedIntList(Py_ssize_t[:] l1, Py_ssize_t[:,:] l2, size_t len_l, size_t len_ll) nogil:
    # The length is passed as an argument, the length o bost lists
    # is assumed to be the same
    #We assume that there is at most one occurence of l1 in l2
    cdef size_t i
    for i in range(len_ll):
        if IntListComparison(l1, l2[i], len_l):
            return i
    return -1

#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef bool IntListComparison(Py_ssize_t[:] l1, Py_ssize_t[:] l2, size_t length) nogil:
    """The length is passed as an argument, the length of both lists
    is assumed to be the same

    Arguments
    ---------
    l1 : C-array memoryview of signed indexing integers of dimension 1

    l2 : C-array memoryview of signed indexing integers of dimension 1 

    length : size_t (unsigned indexing integer) 
        length of both arrays

    Returns
    -------
    True if all elements are the same. False otherwise
    """
    cdef size_t i
    for i in range(length):
        if l1[i] != l2[i] :
            return False 
    return True

#Optimized code (no boundchecks)
#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef vector[vector[vector[double]]] cgetQuads(int[:,:] faces_mv,
               size_t[:] fs, double[:,:] verts_mv, int[:] verts_label_map_mv) nogil:
    cdef: 
        size_t idx, i, j
        vector[vector[vector[double]]] vquads
        vector[vector[double]] face 
        vector[double] coord
    vquads.clear()
    for i in range(fs[0]):
        face.clear()
        for j in range(fs[1]):
            coord.clear()
            idx = idxFirstValue(verts_label_map_mv,faces_mv[i][j])
            for p in range(3):   
                coord.push_back(verts_mv[idx][p])
            face.push_back(coord)
        vquads.push_back(face)
    return vquads

#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void intra_face_distance_matrix(double[:,:] quad_face, SymmetricMatrix m) nogil:
    cdef :
        size_t i, j 
        double[:] pt1, pt2
    for i in range(4):
        for j in range(i,4):
            if i == j :
                m.set_item_nogil(i, j, 0.0)
            else : 
                pt1 = quad_face[i,:]
                pt2 = quad_face[j,:]
                m.set_item_nogil(i, j, distance3D(pt1, pt2))

#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef int idxFirstValue(int[:] mV, int val) nogil:
    cdef Py_ssize_t idx 
    cdef Py_ssize_t i = 0
    cdef bool found = False
    while found == False and i < mV.shape[0]:
        if mV[i] == val :
           idx = i
           found = True
        else :
            i = i+1
    return idx

#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double distance3D(double[:] pt1, double[:] pt2) nogil:
    """Returns distance between  points in 3D coordinates
    """
    return sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)

#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef vector[double] cross_product(double[:] vect1,double[:] vect2):
    """Cross product between 2 3D memoryviews (x1, y1, z1), (x2, y2, z2)
    """
    cdef double x_cross, y_cross, z_cross 
    cdef vector[double] cross_prod 

    x_cross = vect1[1] * vect2[2] - vect1[2] * vect2[1]
    y_cross = -1*(vect1[0] * vect2[2] - vect2[0] * vect1[2])
    z_cross = vect1[0] * vect2[1] - vect1[1] * vect2[0]
    cross_prod.clear()
    cross_prod.push_back(x_cross)
    cross_prod.push_back(y_cross)
    cross_prod.push_back(z_cross)
    return cross_prod

#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef vector[double] cross_product_vect(vector[double] vect1, vector[double] vect2) nogil:
    """Cross product between 2 3D vectors (x1, y1, z1), (x2, y2, z2)
    """
    cdef double x_cross, y_cross, z_cross 
    cdef vector[double] cross_prod 

    x_cross = vect1[1] * vect2[2] - vect1[2] * vect2[1]
    y_cross = -1*(vect1[0] * vect2[2] - vect2[0] * vect1[2])
    z_cross = vect1[0] * vect2[1] - vect1[1] * vect2[0]
    cross_prod.clear()
    cross_prod.push_back(x_cross)
    cross_prod.push_back(y_cross)
    cross_prod.push_back(z_cross)
    return cross_prod

#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef vector[double] quad_centroid(double[:,:] quad) nogil:
    """Centroid of quadrilateral element of shape (4,3)
    """
    cdef double x_sum, y_sum, z_sum 
    cdef vector[double] centroid 
    x_sum = (quad[0,0] + quad[1,0] + quad[2,0] + quad[3,0])/4 
    y_sum = (quad[0,1] + quad[1,1] + quad[2,1] + quad[3,1])/4 
    z_sum = (quad[0,2] + quad[1,2] + quad[2,2] + quad[3,2])/4 
    centroid.clear()
    centroid.push_back(x_sum)
    centroid.push_back(y_sum)
    centroid.push_back(z_sum)
    return centroid

#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef vector[vector[double]] quadCenterEdgeVectors(double[:,:] quad, vector[double] centroid) nogil:
    """list of vectors going from the centroid of the element to the edges.
    if the centroid is (xc,yc,zc) and the quad is
        ((x1, y1, z1),
         (x2, y2, z2),
         (x3, y3, z3),
         (x4, y4, z4)), 

     then the vector is:
         ((xc-x1, yc-y1, zc-z1),
          (xc-x2, yc-y2, zc-z2),
          (xc-x3, yc-y3, zc-z3),
          (xc-x4, yc-y4, zc-z4))
    """
    cdef size_t i, j
    cdef vector[vector[double]] centerEdgeVectors 
    cdef vector[double] edgeVector
    cdef double difference
    centerEdgeVectors.clear()
    for i in range(4):
        edgeVector.clear()
        for j in range(3):
            difference = quad[i,j] - centroid[j]
            edgeVector.push_back(difference)
        centerEdgeVectors.push_back(edgeVector)
    return centerEdgeVectors

#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef vector[double] quadNormal(double[:,:] quad) nogil:
    """For each successive centroid-edge vector calculated above, we calculate the associated
    normal using the cross product. Then we avarage the four normals to get the approximated
    closest normal.
    """
    cdef Py_ssize_t i, j 
    cdef vector[double] centroid
    cdef vector[vector[double]] centerEdgeVectors
    cdef vector[vector[double]] quadEdge2EdgeNormals
    cdef vector[double] _vect, _vect_normed, sum_normals, sum_normals_unit


    centroid = quad_centroid(quad)
    centerEdgeVectors = quadCenterEdgeVectors(quad, centroid)
    sum_normals.clear()
    sum_normals.resize(3)
    for i in range(4):
        if i < 3 :
            j = i+1 
        else:
            j = 0
        _vect = cross_product_vect(centerEdgeVectors[i], centerEdgeVectors[j])
        _vect_normed = vector2unit(_vect)
        sum_normals = sum_vectors(sum_normals, _vect_normed)
    sum_normals_unit = vector2unit(sum_normals)
    return sum_normals_unit

#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef vector[double] vector2unit(vector[double] vect) nogil:
    """norm of n dimensional vector
    """
    cdef vector[double] unitVector
    cdef size_t _size, i 
    _size = vect.size()
    norm = norm_vector(vect)
    unitVector.clear()
    for i in range(_size):
        unitVector.push_back(vect[i]/norm)
    return unitVector

#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double norm_vector(vector[double] vect) nogil:
    """norm of n dimensional vector
    """
    cdef double squared_sum 
    cdef size_t _size, i 
    _size = vect.size()
    squared_sum = .0
    for i in range(_size):
        squared_sum += vect[i]**2
    return sqrt(squared_sum)

#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef vector[double] sum_vectors(vector[double] vect1, vector[double] vect2) nogil:
    """returns the vector sum of two vectors
    """
    cdef vector[double] sum_vects 
    cdef size_t i, N 
    sum_vects.clear()
    N = vect1.size()
    for  i in range(N):
        sum_vects.push_back(vect1[i]+vect2[i])
    return sum_vects

#############################################################################


cdef class SymmetricMatrix:
    # modded from https://blog.sopticek.net/2016/09/18/speeding-up-symmetric-matrix-with-cython/
    cdef size_t _size, _data_size
    cdef vector[double] _data

    def __init__(self, size_t size):
        self._data_size = (size + 1) * size // 2
        self._size = size
        self._data.clear()
        self._data.resize(self._data_size)
        for i in range(self._data_size):
            self._data[i] = .0

    def __len__(self):
        return self._size

    def __setitem__(self, position, value):
        cdef size_t index, row, column 
        row, column = position[0], position[1]
        index = self._get_index(row, column)
        if index >= self._data_size:
            raise IndexError( 'out of bounds')
        self._data[index] = value

    def __getitem__(self, position):
        cdef size_t index, row, column 
        row, column = position[0], position[1]
        index = self._get_index(row, column)
        if index >= self._data_size:
            raise IndexError('out of bounds')
        return self._data[index]

    def argmax(self):
        return self.cargmax()

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef size_t set_item_nogil(self, size_t row, size_t column, double value) nogil:
        cdef size_t index
        index = self._get_index(row, column)
        self._data[index] = value

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef size_t _get_index(self,size_t row, size_t column) nogil:
        cdef size_t index
        if column > row:
            row, column = column, row
        index = (row) * (row + 1) // 2 + column
        return index

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef vector[size_t] _get_position(self, size_t index) nogil:
        cdef vector[size_t] position 
        cdef size_t i, j, row, column, idx_pos 
        position.clear()
        for i in range(self._size):
            for j in range(i+1):
                idx_pos = (i) * (i + 1) // 2 + j
                if idx_pos == index:
                    row = i
                    column = j
                    break
        position.push_back(row)
        position.push_back(column)
        return position

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef vector[size_t] cargmax(self) nogil:
        cdef vector[size_t] position_max
        cdef size_t index_max  
        index_max = argmax(self._data)
        position_max = self._get_position(index_max)
        return position_max
            
#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef size_t argmax(vector[double] vector_cpp) nogil:
    cdef size_t i, idx_max, N 
    cdef double val_max
    N = vector_cpp.size()
    if N > 1 :
        val_max = vector_cpp[0]
        idx_max = 0
        for i in range(1,N):
            if vector_cpp[i] > val_max:
                idx_max = i
                val_max = vector_cpp[i]
        return idx_max
    else :
        return 0


