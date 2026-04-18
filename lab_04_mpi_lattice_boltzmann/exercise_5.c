/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
// Goal: Implement 2D grid communication scheme with
//      8 neighbors using MPI types for non contiguous
//      side.
//
// SUMMARY:
//     - 2D splitting along X and Y
//     - 8 neighbors communications
//     - Blocking communications
// NEW:
//     - >>> MPI type for non contiguous cells <<<
//
//////////////////////////////////////////////////////

/****************************************************/
#include "src/lbm_struct.h"
#include "src/exercises.h"

/****************************************************/
void lbm_comm_init_ex5(lbm_comm_t * comm, int total_width, int total_height)
{
	lbm_comm_init_ex4(comm, total_width, total_height);

	MPI_Datatype rowtype;
	MPI_Type_vector(comm->width, DIRECTIONS, comm->height * DIRECTIONS, MPI_DOUBLE, &rowtype);
	MPI_Type_commit(&rowtype);
	comm->type = rowtype;
}

/****************************************************/
void lbm_comm_release_ex5(lbm_comm_t * comm)
{
	lbm_comm_release_ex4(comm);

	MPI_Type_free(&comm->type);
}

/****************************************************/
void lbm_comm_ghost_exchange_ex5(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	int rank_up = comm->rank_y == 0 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x, comm->rank_y - 1);

	int rank_right = comm->rank_x == comm->nb_x - 1 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x + 1, comm->rank_y);

	int rank_down = comm->rank_y == comm->nb_y - 1 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x, comm->rank_y + 1);

	int rank_left = comm->rank_x == 0 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x - 1, comm->rank_y);

	int count_data_vertically = comm->height * DIRECTIONS;
	
	//right 
	MPI_Sendrecv(lbm_mesh_get_cell(mesh,comm->width-2, 0), count_data_vertically, MPI_DOUBLE, rank_right, TAG, lbm_mesh_get_cell(mesh,0,0), count_data_vertically, MPI_DOUBLE, rank_left, TAG, comm->communicator, MPI_STATUS_IGNORE);
	
	//left 
	MPI_Sendrecv(lbm_mesh_get_cell(mesh,1,0), count_data_vertically, MPI_DOUBLE, rank_left, TAG, lbm_mesh_get_cell(mesh,comm->width-1,0), count_data_vertically, MPI_DOUBLE, rank_right, TAG, comm->communicator, MPI_STATUS_IGNORE);

	//up and down
	MPI_Sendrecv(lbm_mesh_get_cell(mesh, 0, 1), 1, comm->type, rank_up, TAG, lbm_mesh_get_cell(mesh, 0, comm->height - 1), 1, comm->type, rank_down, TAG, comm->communicator, MPI_STATUS_IGNORE);
	MPI_Sendrecv(lbm_mesh_get_cell(mesh, 0, comm->height - 2), 1, comm->type, rank_down, TAG, lbm_mesh_get_cell(mesh, 0, 0), 1, comm->type, rank_up, TAG, comm->communicator, MPI_STATUS_IGNORE);	
}


//TODO: Send output to the 0th rank.