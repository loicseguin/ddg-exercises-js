"use strict";

/**
 * @module Projects
 */
class SimplicialComplexOperators {

        /** This class implements various operators (e.g. boundary, star, link) on a mesh.
         * @constructor module:Projects.SimplicialComplexOperators
         * @param {module:Core.Mesh} mesh The input mesh this class acts on.
         * @property {module:Core.Mesh} mesh The input mesh this class acts on.
         * @property {module:LinearAlgebra.SparseMatrix} A0 The vertex-edge adjacency matrix of <code>mesh</code>.
         * @property {module:LinearAlgebra.SparseMatrix} A1 The edge-face adjacency matrix of <code>mesh</code>.
         */
        constructor(mesh) {
                this.mesh = mesh;
                this.assignElementIndices(this.mesh);

                this.A0 = this.buildVertexEdgeAdjacencyMatrix(this.mesh);
                this.A1 = this.buildEdgeFaceAdjacencyMatrix(this.mesh);
        }

        /** Assigns indices to the input mesh's vertices, edges, and faces
         * @method module:Projects.SimplicialComplexOperators#assignElementIndices
         * @param {module:Core.Mesh} mesh The input mesh which we index.
         */
        assignElementIndices(mesh) {
          for (let i = 0; i < mesh.vertices.length; i++) {
            mesh.vertices[i].index = i;
          }
          for (let i = 0; i < mesh.edges.length; i++) {
            mesh.edges[i].index = i;
          }
          for (let i = 0; i < mesh.faces.length; i++) {
            mesh.faces[i].index = i;
          }
        }

        /** Returns the vertex-edge adjacency matrix of the given mesh.
         * @method module:Projects.SimplicialComplexOperators#buildVertexEdgeAdjacencyMatrix
         * @param {module:Core.Mesh} mesh The mesh whose adjacency matrix we compute.
         * @returns {module:LinearAlgebra.SparseMatrix} The vertex-edge adjacency matrix of the given mesh.
         */
        buildVertexEdgeAdjacencyMatrix(mesh) {
          let T = new Triplet(mesh.edges.length, mesh.vertices.length);

          for (let edge of mesh.edges) {
            let vi = edge.halfedge.vertex.index;
            let vj = edge.halfedge.twin.vertex.index;
            T.addEntry(1, edge.index, vi);
            T.addEntry(1, edge.index, vj);
          }

          return SparseMatrix.fromTriplet(T);
        }

        /** Returns the edge-face adjacency matrix.
         * @method module:Projects.SimplicialComplexOperators#buildEdgeFaceAdjacencyMatrix
         * @param {module:Core.Mesh} mesh The mesh whose adjacency matrix we compute.
         * @returns {module:LinearAlgebra.SparseMatrix} The edge-face adjacency matrix of the given mesh.
         */
        buildEdgeFaceAdjacencyMatrix(mesh) {
          let T = new Triplet(mesh.faces.length, mesh.edges.length);

          for (let face of mesh.faces) {
            for (let edge of face.adjacentEdges()) {
              T.addEntry(1, face.index, edge.index);
            }
          }

          return SparseMatrix.fromTriplet(T);
        }

        /** Returns a column vector representing the vertices of the
         * given subset.
         * @method module:Projects.SimplicialComplexOperators#buildVertexVector
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:LinearAlgebra.DenseMatrix} A column vector with |V| entries. The ith entry is 1 if
         *  vertex i is in the given subset and 0 otherwise
         */
        buildVertexVector(subset) {
          let vvec = DenseMatrix.zeros(this.mesh.vertices.length, 1);
          for (let i of subset.vertices) {
            vvec.set(1, i, 0);
          }
          return vvec;
        }

        /** Returns a column vector representing the edges of the
         * given subset.
         * @method module:Projects.SimplicialComplexOperators#buildEdgeVector
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:LinearAlgebra.DenseMatrix} A column vector with |E| entries. The ith entry is 1 if
         *  edge i is in the given subset and 0 otherwise
         */
        buildEdgeVector(subset) {
          let evec = DenseMatrix.zeros(this.mesh.edges.length, 1);
          for (let i of subset.edges) {
            evec.set(1, i, 0);
          }
          return evec;
        }

        /** Returns a column vector representing the faces of the
         * given subset.
         * @method module:Projects.SimplicialComplexOperators#buildFaceVector
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:LinearAlgebra.DenseMatrix} A column vector with |F| entries. The ith entry is 1 if
         *  face i is in the given subset and 0 otherwise
         */
        buildFaceVector(subset) {
          let fvec = DenseMatrix.zeros(this.mesh.faces.length, 1);
          for (let i of subset.faces) {
            fvec.set(1, i, 0);
          }
          return fvec;
        }

        /** Returns the star of a subset.
         * @method module:Projects.SimplicialComplexOperators#star
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:Core.MeshSubset} The star of the given subset.
         */
        star(subset) {
          // Careful here: a face is a subset of a simplex. Hence, given a
          // vertex, an adjacent edge is not a face of the vertex. Similarly,
          // a face adjacent to an edge is not a face of the edge.

          // Make a copy so that original is unchanged, i.e., function has no
          // side-effet.
          subset = MeshSubset.deepCopy(subset);

          // Edges adjacent to vertices.
          let vvec = this.buildVertexVector(subset);
          let B = this.A0.timesDense(vvec);
          for (let i = 0; i < B.nRows(); i++) {
            if (B.get(i, 0) > 0) subset.addEdge(i);
          }

          // Faces adjacent to edges. Edges adjacent to vertices are already
          // added, so this is enough.
          let evec = this.buildEdgeVector(subset);
          B = this.A1.timesDense(evec);
          for (let i = 0; i < B.nRows(); i++) {
            if (B.get(i, 0) > 0) subset.addFace(i);
          }

          return subset;
        }

        /** Returns the closure of a subset.
         * @method module:Projects.SimplicialComplexOperators#closure
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:Core.MeshSubset} The closure of the given subset.
         */
        closure(subset) {
          // Make a copy so that original is unchanged, i.e., function has no
          // side-effet.
          subset = MeshSubset.deepCopy(subset);

          // Edges adjacent to faces.
          let fvec = this.buildFaceVector(subset);
          let B = this.A1.transpose().timesDense(fvec);
          for (let i = 0; i < B.nRows(); i++) {
            if (B.get(i, 0) > 0) subset.addEdge(i);
          }

          // Vertices adjacent to edges. Edges adjacent to faces are already
          // included, so this is enough.
          let evec = this.buildEdgeVector(subset);
          B = this.A0.transpose().timesDense(evec);
          for (let i = 0; i < B.nRows(); i++) {
            if (B.get(i, 0) > 0) subset.addVertex(i);
          }

          return subset;
        }

        /** Returns the link of a subset.
         * @method module:Projects.SimplicialComplexOperators#link
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:Core.MeshSubset} The link of the given subset.
         */
        link(subset) {
          let closure_of_star = this.closure(this.star(subset));
          let star_of_closure = this.star(this.closure(subset));
          closure_of_star.deleteSubset(star_of_closure);

          return closure_of_star;
        }

        /** Returns true if the given subset is a subcomplex and false otherwise.
         * @method module:Projects.SimplicialComplexOperators#isComplex
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {boolean} True if the given subset is a subcomplex and false otherwise.
         */
        isComplex(subset) {
          let closure = this.closure(subset);
          return closure.equals(subset);
        }

        /** Returns the degree if the given subset is a pure subcomplex and -1 otherwise.
         * @method module:Projects.SimplicialComplexOperators#isPureComplex
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {number} The degree of the given subset if it is a pure subcomplex and -1 otherwise.
         */
        isPureComplex(subset) {
          if (!this.isComplex(subset)) {
            return -1;
          }

          if (subset.faces.size > 0) {
            // Subset contains faces, hence there should be no hanging edges or
            // vertices for it to be pure 2-complex.
            // Hanging vertices? Since an hanging edge also has vertices not
            // adjacent to faces, this is sufficient.
            let A = (this.A1.timesSparse(this.A0)).transpose();
            let fvec = this.buildFaceVector(subset);
            let vvec = A.timesDense(fvec);
            for (let vi of subset.vertices) {
              if (vvec.get(vi, 0) == 0) {
                return -1;
              }
            }
            return 2;
          } else if (subset.edges.size > 0) {
            // Subset contains edges, hence there should be no hanging vertices
            // for it to be pure 1-complex.
            // Hanging vertices?
            let evec = this.buildEdgeVector(subset);
            let vvec = this.A0.transpose().timesDense(evec);
            for (let vi of subset.vertices) {
              if (vvec.get(vi, 0) == 0) {
                return -1;
              }
            }
            return 1;
          }
          return 0;
        }

        /** Returns the boundary of a subset.
         * @method module:Projects.SimplicialComplexOperators#boundary
         * @param {module:Core.MeshSubset} subset A subset of our mesh. We assume <code>subset</code> is a pure subcomplex.
         * @returns {module:Core.MeshSubset} The boundary of the given pure subcomplex.
         */
        boundary(subset) {
          let bound = new MeshSubset();

          if (subset.faces.size > 0) {
            // Find edges that are proper faces of exactly one face.
            let fvec = this.buildFaceVector(subset);
            let evec = this.A1.transpose().timesDense(fvec);
            for (let i = 0; i < evec.nRows(); i++) {
              if (evec.get(i, 0) == 1) {
                bound.addEdge(i);
              }
            }
          } else if (subset.edges.size > 0) {
            // Find vertices that are proper faces of exactly one edge.
            let evec = this.buildEdgeVector(subset);
            let vvec = this.A0.transpose().timesDense(evec);
            for (let i = 0; i < vvec.nRows(); i++) {
              if (vvec.get(i, 0) == 1) {
                bound.addVertex(i);
              }
            }
          }

          return this.closure(bound);
        }
}
