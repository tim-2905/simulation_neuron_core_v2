// header and includes
#include <stdint.h>  // uint32_t
#include <stdbool.h> // bool
#include <math.h>    // powf
#include <stddef.h>  // size_t

// HW I/O
#include <fsl.h>     // AXI-Stream: getfslx / putfslx
#include <xil_io.h>  // AXI-MM: Xil_In32 / Xil_Out32
#include <riscv_interface.h> // fsl_isinvalid(result)

// constants
#define NUM_NEURONS 1024
#define DT          1.0f

static float tau_mem_inv     = 0.2f;
static float spike_threshold = 1.0f;

// HW base addresses / IDs  
#define BRAM_BASE  0xC0000000      // axi_bram_ctrl_0 
#define GPIO_BASE_current_timestep  0x40010000      // axi_gpio_0, Channel-1 Data 
#define RX_ID      0                // S0_AXIS 
#define TX_ID      0                // M0_AXIS 
#define GPIO_while_loop_flag_BASE 0x40030000
#define GPIO_busy_flag_BASE 0x40000000
#define GPIO_timestep_done_BASE 0x40020000
#define GPIO_BASE_BRAM_flag 0x40040000
// #define NO_DATA_MARKER 0xFFFFFFFE



// states 
static float    neuron_states[NUM_NEURONS] = {0}; 
static uint32_t last_timesteps[NUM_NEURONS] = {0}; 


uint32_t riscv_busy = 0;

// prototypes 
// static uint32_t hot_fifo_pop(void);
static float    weight_sum_read(uint32_t neuron_id);
static void    output_spiking(uint32_t neuron_id, float v_new, float v_old);
static void    LIF_update_one(uint32_t current_timestep, uint32_t neuron_id);
static inline void    run_while_loop_flag(uint32_t while_loop_flag);
static inline void    send_busy_flag(uint32_t riscv_busy);
static bool pop_hot_neurons_tvalid(uint32_t *w);
static uint32_t send_timestep_done_flag(uint32_t timestep_done);


// pop hot neurons tvalid flag
static bool pop_hot_neurons_tvalid(uint32_t *w) {

//    *w = NO_DATA_MARKER;
  
//    getfslx(*w, RX_ID, FSL_NONBLOCKING);

//    if (*w == NO_DATA_MARKER) {
//        return false;
//    }

//    return true;
//}

// new approach 


  getfslx(*w, RX_ID, FSL_NONBLOCKING);
    
	uint32_t isinvalid = 0;
	fsl_isinvalid(isinvalid);
	
	if(isinvalid != 0){
		
		return false;
		
	}
	
	return true;
}

// send BRAM_flag

static uint32_t send_BRAM(uint32_t BRAM_1, uint32_t BRAM_2){
	
	Xil_Out32(GPIO_BASE_BRAM_flag, BRAM_1);
	Xil_Out32(GPIO_BASE_BRAM_flag, BRAM_2);
	
}


// send timeste_done flag
static uint32_t send_timestep_done_flag(uint32_t timestep_done){
	
	Xil_Out32(GPIO_timestep_done_BASE, timestep_done);
	return timestep_done;
}

// send run while loop flag
static inline void run_while_loop_flag(uint32_t while_loop_flag){
	
	Xil_Out32(GPIO_while_loop_flag_BASE, while_loop_flag);
	
}

// send risc v busy flag
static inline void send_busy_flag(uint32_t riscv_busy){
	
	Xil_Out32(GPIO_busy_flag_BASE, riscv_busy);
	
}


// Float<->U32 helpers; Bit-Reinterpretation für Stream/BRAM
static inline uint32_t f2u(float f){ union { float f; uint32_t u; } v = { .f = f }; return v.u; }
static inline float    u2f(uint32_t u){ union { float f; uint32_t u; } v = { .u = u }; return v.f; }

// AXI-GPIO: current_timestep von TB lesen
static inline uint32_t read_current_timestep(void){
    return Xil_In32(GPIO_BASE_current_timestep + 0x0);  // Channel-1 Data
}

// AXI-MM: 32-bit-Wort an Wort-Index lesen 
static inline uint32_t bram_read_u32_at_index(uint32_t index){
    return Xil_In32(BRAM_BASE + 4 * index);  // 32-bit Wort -> Byteoffset = 4 * index
}

// ID->Index-Mapping
static inline uint32_t weight_index_from_neuron_id(uint32_t neuron_id){
    return neuron_id;  
}

// Gewicht lesen: BRAM enthält IEEE-754-Floats 
static inline float weight_sum_read(uint32_t neuron_id){
    uint32_t idx = weight_index_from_neuron_id(neuron_id);
    uint32_t raw = bram_read_u32_at_index(idx);
	send_BRAM(raw, idx);
    return u2f(raw);  // Float-Bitmuster -> float
}

// AXI-Stream: Spike-Ausgabe (ID, v_new, v_old , alle als Bitmuster)
static void output_spiking(uint32_t neuron_id, float v_new, float v_old){
    putfslx(neuron_id,  TX_ID, FSL_DEFAULT);  // Wort 1: ID
    putfslx(f2u(v_new), TX_ID, FSL_DEFAULT);  // Wort 2: v_new
    putfslx(f2u(v_old), TX_ID, FSL_DEFAULT);  // Wort 3: v_old
}

// AXI-Stream: Hot-Neuron-ID lesen 
//static uint32_t hot_fifo_pop(void){
//    uint32_t w;
//    getfslx(w, RX_ID, FSL_DEFAULT);  // neuron_id oder END_MARKER
//    return w;
// }


// LIF update logic 

static void LIF_update_one(uint32_t current_timestep, uint32_t neuron_id)
{
	riscv_busy++;
	send_busy_flag(riscv_busy);
	
    float v_old = neuron_states[neuron_id];
    uint32_t t_last = last_timesteps[neuron_id];
    uint32_t delta_time = current_timestep - t_last;
    float current = weight_sum_read(neuron_id);
    float current_scaled = current * tau_mem_inv;
    float v_leak = powf(1.0f - tau_mem_inv, (float)delta_time) * v_old;
    float v_new = v_leak + current_scaled;

    if (v_new >= spike_threshold){
         
		output_spiking(neuron_id, v_new, v_old); 
		v_new = 0.0f;
		
    }

    neuron_states[neuron_id]  = v_new;
    last_timesteps[neuron_id] = current_timestep;
	
	
	send_busy_flag(0);
}


// main: lesen & bei END_MARKER beenden

int main(void)
{
	
//	uint32_t ts_current = read_current_timestep();
	bool timestep_done = false;
	
    while(1) {
	
		uint32_t neuron_id;
		run_while_loop_flag(1);
	
//         if (ts_current != read_current_timestep()) { // kann man den wert, den man mit dem funktionsaufruf hier liest auch temporär zwischenspeichern in "ts_temp"?
//		 
//		     ts_current = read_current_timestep(); // ts_temp anstatt read_current_timestep() ?
//		     timestep_done = false;
//		 }

			while (timestep_done == false) {
				

				if (!pop_hot_neurons_tvalid(&neuron_id)){
				// pop hot neurons ( tget ) // tget falls das, was copilot behauptet hat wirklich stimmt
				// if hot neuron fifo was not valid on last pop
					// timestep_done
			   // if not timeste_done
				 // LIF_update
			// end while schleife
			
					timestep_done = true;
					send_timestep_done_flag(timestep_done);
					break;
				}
				
				else{
					
//					        uint32_t neuron_id = hot_fifo_pop();        // Daten aus AXI-Stream
//							if (neuron_id == END_MARKER) break;         // eindeutiges Ende 

							uint32_t now = read_current_timestep();     // Zeit aus TB/AXI-GPIO
							LIF_update_one(now, neuron_id);
//							send_timestep_done_flag(timestep_done);		
		
				}
			}
	   	
		
		if (pop_hot_neurons_tvalid(&neuron_id)){
			
			timestep_done = false;
			send_timestep_done_flag(timestep_done);  
			uint32_t now = read_current_timestep();
			LIF_update_one(now, neuron_id);
			
		}
		
		
    } // end of while(1)-loop
	
} // end of main-loop
