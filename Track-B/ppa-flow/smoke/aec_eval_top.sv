module aec_eval_top (
    input  logic          clk,
    input  logic          rst_n,
    input  logic          load_valid,
    output logic          load_ready,
    input  logic [2:0]    load_target,
    input  logic [31:0]   load_addr,
    input  logic [127:0]  load_data,
    input  logic [15:0]   load_strb,
    input  logic          launch_valid,
    output logic          launch_ready,
    input  logic [31:0]   grid_x,
    input  logic [31:0]   grid_y,
    input  logic [31:0]   grid_z,
    input  logic [31:0]   block_x,
    input  logic [31:0]   block_y,
    input  logic [31:0]   block_z,
    input  logic [31:0]   program_instructions,
    output logic          result_valid,
    input  logic          result_ready,
    output logic [2:0]    result_status,
    output logic [63:0]   result_cycles,
    input  logic          read_valid,
    output logic          read_ready,
    input  logic [31:0]   read_addr,
    output logic          read_data_valid,
    output logic [127:0]  read_data,
    output logic          mem_req_valid,
    input  logic          mem_req_ready,
    output logic          mem_req_write,
    output logic          mem_req_space,
    output logic [31:0]   mem_req_addr,
    output logic [1023:0] mem_req_wdata,
    output logic [127:0]  mem_req_wstrb,
    output logic [3:0]    mem_req_tag,
    input  logic          mem_rsp_valid,
    output logic          mem_rsp_ready,
    input  logic [1023:0] mem_rsp_rdata,
    input  logic [3:0]    mem_rsp_tag,
    input  logic          mem_rsp_error
);

    logic [63:0] state_q;
    logic [63:0] state_d;

    always_comb begin
        state_d = state_q;
        state_d ^= {load_addr, load_data[31:0]};
        state_d ^= {grid_x[15:0], grid_y[15:0], grid_z[15:0], block_x[15:0]};
        state_d ^= {block_y[15:0], block_z[15:0], program_instructions};
        state_d ^= {read_addr, mem_rsp_rdata[31:0]};
        state_d ^= {34'b0, load_target, load_strb, mem_rsp_tag,
                    load_valid, launch_valid, result_ready, read_valid,
                    mem_req_ready, mem_rsp_valid, mem_rsp_error};
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= '0;
        end else begin
            state_q <= state_d;
        end
    end

    always_comb begin
        load_ready = state_q[0];
        launch_ready = state_q[1];
        result_valid = state_q[2];
        result_status = state_q[5:3];
        result_cycles = state_q;
        read_ready = state_q[6];
        read_data_valid = state_q[7];
        read_data = {state_q, ~state_q};
        mem_req_valid = state_q[8];
        mem_req_write = state_q[9];
        mem_req_space = state_q[47];
        mem_req_addr = state_q[41:10];
        mem_req_wdata = {16{state_q}};
        mem_req_wstrb = {2{state_q}};
        mem_req_tag = state_q[45:42];
        mem_rsp_ready = state_q[46];
    end

endmodule
