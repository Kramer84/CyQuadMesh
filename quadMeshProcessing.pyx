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

    def rebaseVertices(self):
        outputs = self.getRebasedVerticesAndLabelMap()
        return outputs


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
        cdef vector[int] basic_indexing 
        cdef vector[int] new_indexing 
        cdef double[:,:] quad_mv
        cdef bool isSame
        cdef int isCooplanar
        #cdef SymmetricMatrix quad_distance_matrix = SymmetricMatrix(4)

        # Intializing indexing 
        basic_indexing.clear()
        new_indexing.clear()
        for i in range(4):
            basic_indexing.push_back(i)

        n_quads = self.quads_mv.shape[0]

        for i in range(n_quads):
            quad_mv = self.quads_mv[i,:,:]
            new_indexing = self.correct_winding_1face(quad_mv) #, quad_distance_matrix
            isCooplanar = self.checkIfFaceCooplanar(quad_mv)

            isSame = IntVectorComparison(new_indexing, basic_indexing, 4)
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
    

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef vector[int] correct_winding_1face(self, double[:,:] quad_face ):
        """Checks the winding for one face and returns the new
        order of the indices.

        Method : 
        - We calculate the centroid 
        - We construct 4 vectors going from the centroid to each edge
        - We calculate the sign of each consecutive cross_product
        - We caculate the sum between each permutation of the cross products, and observe the norm of the resulting vector
        """
        cdef :
            size_t i, j
            int n_are_max = 0 
            double cross_prod_norm
            int[2] permut
            vector[int] basic_indexing = [0,1,2,3]
            vector[int] new_indexing = [0,1,2,3]
            size_t[6][2] index_permut = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
            vector[int] cross_prod_permut_sum_norm_is_max
            vector[double] centroid, cross_prod_sum, cross_prod_norms, cross_prod_permut_sum_norms, max_norm_cross_prod
            vector[vector[double]] center_edge_vectors, consecutive_cross_products, cross_prod_permut_sum

        # Intializing indexing 
        basic_indexing.clear()
        new_indexing.clear()
        for i in range(4):
            basic_indexing.push_back(i)
            new_indexing.push_back(i)

        # Clearing C++ vectors
        centroid.clear()
        center_edge_vectors.clear()
        consecutive_cross_products.clear()
        

        centroid = quad_centroid(quad_face)
        center_edge_vectors = quadCenterEdgeVectors(quad_face, centroid)
        #Here we will have the 4 consecutive cross products: Element 0 and 1 , 1 and 2, 2 and 3 and 3 and 0
        for i in range(3):
            consecutive_cross_products.push_back(cross_product_vect(center_edge_vectors[i],center_edge_vectors[i+1]))
        # Here for cross product 3 to 0
        consecutive_cross_products.push_back(cross_product_vect(center_edge_vectors[3],center_edge_vectors[0]))

        cross_prod_norms.clear()
        for i in range(4):
            cross_prod_norms.push_back(norm_vector(consecutive_cross_products[i]))

        #Now calculating the sums between the cross products:
        cross_prod_permut_sum.clear()
        cross_prod_permut_sum_norms.clear()
        max_norm_cross_prod.clear()
        cross_prod_permut_sum_norm_is_max.clear()
        for i in range(6):
            permut = index_permut[i]

            max_norm_cross_prod.push_back(max_doubles(cross_prod_norms[permut[0]], 
                                                      cross_prod_norms[permut[1]]))

            cross_prod_sum = sum_vectors(consecutive_cross_products[permut[0]], 
                                         consecutive_cross_products[permut[1]])

            cross_prod_permut_sum.push_back(cross_prod_sum) #Sum of the cross products
            cross_prod_permut_sum_norms.push_back(norm_vector(cross_prod_sum)) #Norm of the sum of the cross products

            if cross_prod_permut_sum_norms[i] > max_norm_cross_prod[i]:
                cross_prod_permut_sum_norm_is_max.push_back(1)
                n_are_max += 1

            else : 
                cross_prod_permut_sum_norm_is_max.push_back(0)
                n_are_max += 0

        if  n_are_max == 6 :
            return new_indexing

        else : 
            print('\t\t\tWinding not OKKKK')
            return new_indexing


    #########################################################################

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef getQuads(self):
        cdef size_t I, J 
        I = self.faces_mv.shape[0]
        J = self.faces_mv.shape[1] 
        cdef np.ndarray[np.float64_t, ndim=3] nquads = np.zeros(shape=(I, J, 3), dtype = "float64")
        cdef size_t i, j, p

        self.vquads.clear()
        self.vquads = cgetQuads(self.faces_mv, I, J, self.verts_mv, self.verts_label_map_mv)
        for i in range(I):
            for j in range(J):
                for p in range(3):
                    nquads[i,j,p] = self.vquads[i][j][p]
        self.vquads.clear()
        self.quads_mv = nquads.astype(dtype="float64", subok=True, copy=False)
        return nquads  

    #########################################################################
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef getRebasedVerticesAndLabelMap(self):
        """This function is because we can build the mesh by passing a list of vertices
        having more vertices then those used in faces. Herewe drop all
        unused vertices, and return a reconstructed label map and vertex array
        """
        cdef size_t i, n_vertices
        cdef int idx, lbl

        cdef np.ndarray[int, ndim=1] unique_vertex_labels = np.unique(np.ndarray.flatten(self.faces.astype('int32')))[ ~ np.isnan(np.unique(np.ndarray.flatten(self.faces.astype('int32')))) ]

        n_vertices = unique_vertex_labels.shape[0]

        cdef np.ndarray[double, ndim=2] rebased_vertices = np.zeros(shape=(n_vertices, 3), dtype = "float64")
        cdef np.ndarray[int, ndim=1] new_label_map = np.zeros(shape=(n_vertices, ), dtype = "int32")

        for i in range(n_vertices):
                lbl = unique_vertex_labels[i] # Label of vertex
                idx = np.argwhere(self.vertices_label_map == lbl)[0]     # Idx if vertex
                new_label_map[i] = lbl
                rebased_vertices[i,:] = self.vertices[idx,:]

        return rebased_vertices, new_label_map

#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef int IntListInNestedIntList(int[:] l1, int[:,:] l2, size_t len_l, size_t len_ll) nogil:
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
cdef bool IntListComparison(int[:] l1, int[:] l2, size_t length) nogil:
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

#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef bool IntVectorComparison(vector[int] V1, vector[int] V2, size_t length) nogil:
    """The length is passed as an argument, the length of both lists
    is assumed to be the same

    Arguments
    ---------
    l1 : C++ int vetor

    l2 : C++ int vetor

    length : size_t (unsigned indexing integer) 
        length of both vectors

    Returns
    -------
    True if all elements are the same. False otherwise
    """
    cdef size_t i
    for i in range(length):
        if V1[i] != V2[i] :
            return False 
    return True

#Optimized code (no boundchecks)
#############################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef vector[vector[vector[double]]] cgetQuads(int[:,:] faces_mv, size_t I, 
            size_t J, double[:,:] verts_mv, int[:] verts_label_map_mv) nogil:
    cdef: 
        size_t idx, i, j
        vector[vector[vector[double]]] vquads
        vector[vector[double]] face 
        vector[double] coord
    vquads.clear()
    for i in range(I):
        face.clear()
        for j in range(J):
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
    """returns a normalized n dimensional vector of sze 1 
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

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double max_doubles(double X0, double X1) nogil:
    """Function that returrns the maximum value between two doubles
    """
    if X0 >= X1:
        return X0
    else : 
        return X1
