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
//       8 neighbors using manual copy for non
//       contiguous side and blocking communications
//
// SUMMARY:
//     - 2D splitting along X and Y
//     - 8 neighbors communications
//     - Blocking communications
//     - Manual copy for non continguous cells
//
//////////////////////////////////////////////////////

/****************************************************/
#include "src/lbm_struct.h"
#include "src/exercises.h"
#include <assert.h>

/****************************************************/
void lbm_comm_init_ex4(lbm_comm_t * comm, int total_width, int total_height)
{
	int size, rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    int periods[2] = {0, 0};
    int reorder = 0;
    MPI_Comm cart_comm = NULL;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
	if(cart_comm == NULL) fatal("Error whilst creating the cartesian communication");

    int coords[2];
    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

	comm->communicator = cart_comm;

	comm->nb_x = dims[0]; 
	if(total_width % comm->nb_x != 0) warning("number of processes on the X-axis doesn't divide the total width");
	comm->nb_y = dims[1]; 
	if(total_width % comm->nb_y != 0) warning("number of processes on the Y-axis doesn't divide the total width");

	comm->rank_x = coords[0];
	comm->rank_y = coords[1];

	comm->width = (total_width / comm->nb_x) + 2;
	comm->height = (total_height / comm->nb_y) + 2;

	comm->x = comm->rank_x * (total_width / comm->nb_x);
	comm->y = comm->rank_y * (total_height / comm->nb_y);

	size_t alloc_space = (comm->width) * DIRECTIONS * sizeof(double);

	comm->buffer_send_up = malloc(alloc_space);
	assert(comm->buffer_send_up);

	comm->buffer_send_down = malloc(alloc_space);
	assert(comm->buffer_send_down);

	comm->buffer_recv_up = malloc(alloc_space);
	assert(comm->buffer_recv_up);

	comm->buffer_recv_down = malloc(alloc_space);
	assert(comm->buffer_recv_down);

	comm->req_count = 0;

	lbm_comm_print(comm);
}

/****************************************************/
void lbm_comm_release_ex4(lbm_comm_t * comm)
{
	MPI_Comm_free(&comm->communicator);
	free(comm->buffer_recv_down);
	free(comm->buffer_recv_up);
	free(comm->buffer_send_down);
	free(comm->buffer_send_up);
}

/****************************************************/
void lbm_comm_ghost_exchange_ex4(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	//annoying fucking boundary calculations 
	int rank_up = comm->rank_y == 0 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x, comm->rank_y - 1);

	int rank_right = comm->rank_x == comm->nb_x - 1 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x + 1, comm->rank_y);

	int rank_down = comm->rank_y == comm->nb_y - 1 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x, comm->rank_y + 1);

	int rank_left = comm->rank_x == 0 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x - 1, comm->rank_y);

	int count_data_vertically = comm->height * DIRECTIONS;

	//right 
	MPI_Sendrecv(lbm_mesh_get_cell(mesh,comm->width-2, 0), count_data_vertically, MPI_DOUBLE, rank_right, TAG, lbm_mesh_get_cell(mesh,0,0), count_data_vertically, MPI_DOUBLE, rank_left, TAG, comm->communicator, MPI_STATUS_IGNORE);
	
	//left 
	MPI_Sendrecv(lbm_mesh_get_cell(mesh,1,0), count_data_vertically, MPI_DOUBLE, rank_left, TAG, lbm_mesh_get_cell(mesh,comm->width-1,0), count_data_vertically, MPI_DOUBLE, rank_right, TAG, comm->communicator, MPI_STATUS_IGNORE);
	
	manual_copy_to_buffer_x_axis(1, comm->buffer_send_up, comm, mesh);
	manual_copy_to_buffer_x_axis(comm->height - 2, comm->buffer_send_down, comm, mesh);

	//up and down
	MPI_Sendrecv(comm->buffer_send_up, (comm->width) * DIRECTIONS, MPI_DOUBLE, rank_up, TAG, comm->buffer_recv_down, (comm->width) * DIRECTIONS, MPI_DOUBLE, rank_down, TAG, comm->communicator, MPI_STATUS_IGNORE);
	MPI_Sendrecv(comm->buffer_send_down, (comm->width) * DIRECTIONS, MPI_DOUBLE, rank_down, TAG, comm->buffer_recv_up, (comm->width) * DIRECTIONS, MPI_DOUBLE, rank_up, TAG, comm->communicator, MPI_STATUS_IGNORE);

	if(rank_up != MPI_PROC_NULL)
		manual_copy_buffer_to_mesh_x_axis(0, comm->buffer_recv_up, comm, mesh);
 
	if(rank_down != MPI_PROC_NULL) 
		manual_copy_buffer_to_mesh_x_axis(comm->height - 1, comm->buffer_recv_down, comm, mesh);

}
