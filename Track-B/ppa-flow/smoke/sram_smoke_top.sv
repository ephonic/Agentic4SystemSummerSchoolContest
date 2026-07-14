module sram_smoke_top (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         en,
    input  logic         read_en,
    input  logic         write_en,
    input  logic [9:0]   addr,
    input  logic [63:0]  write_data,
    output logic [319:0] read_data
);
    logic [31:0] direct_256x32;
    logic [31:0] direct_512x32;
    logic [31:0] direct_1024x32;
    logic [63:0] direct_256x64;
    logic [31:0] wrapped_256x32;
    logic [31:0] wrapped_512x32;
    logic [31:0] wrapped_1024x32;
    logic [63:0] wrapped_256x64;

    srambank_64x4x32_6t122 u_direct_256x32 (
        .clk(clk), .ADDRESS(addr[7:0]), .wd(write_data[31:0]),
        .banksel(en), .read(read_en), .write(write_en), .dataout(direct_256x32)
    );
    srambank_128x4x32_6t122 u_direct_512x32 (
        .clk(clk), .ADDRESS(addr[8:0]), .wd(write_data[31:0]),
        .banksel(en), .read(read_en), .write(write_en), .dataout(direct_512x32)
    );
    srambank_256x4x32_6t122 u_direct_1024x32 (
        .clk(clk), .ADDRESS(addr), .wd(write_data[31:0]),
        .banksel(en), .read(read_en), .write(write_en), .dataout(direct_1024x32)
    );
    srambank_64x4x64_6t122 u_direct_256x64 (
        .clk(clk), .ADDRESS(addr[7:0]), .wd(write_data),
        .banksel(en), .read(read_en), .write(write_en), .dataout(direct_256x64)
    );

    aec_sram_256x32 u_wrapped_256x32 (
        .clk(clk), .en(en), .read_en(read_en), .write_en(write_en),
        .addr(addr[7:0]), .write_data(write_data[31:0]), .read_data(wrapped_256x32)
    );
    aec_sram_512x32 u_wrapped_512x32 (
        .clk(clk), .en(en), .read_en(read_en), .write_en(write_en),
        .addr(addr[8:0]), .write_data(write_data[31:0]), .read_data(wrapped_512x32)
    );
    aec_sram_1024x32 u_wrapped_1024x32 (
        .clk(clk), .en(en), .read_en(read_en), .write_en(write_en),
        .addr(addr), .write_data(write_data[31:0]), .read_data(wrapped_1024x32)
    );
    aec_sram_256x64 u_wrapped_256x64 (
        .clk(clk), .en(en), .read_en(read_en), .write_en(write_en),
        .addr(addr[7:0]), .write_data(write_data), .read_data(wrapped_256x64)
    );

    always_comb begin
        read_data = {direct_256x32, direct_512x32, direct_1024x32,
                     direct_256x64, wrapped_256x32, wrapped_512x32,
                     wrapped_1024x32, wrapped_256x64};
    end

    logic unused_rst_n;
    assign unused_rst_n = rst_n;
endmodule
