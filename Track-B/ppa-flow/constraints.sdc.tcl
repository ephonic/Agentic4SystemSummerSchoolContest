set_cmd_units -time ns -capacitance ff

set period $::env(PPA_CLOCK_PERIOD_NS)
set uncertainty [expr {$period * $::env(PPA_UNCERTAINTY_RATIO)}]
set io_delay [expr {$period * $::env(PPA_IO_DELAY_RATIO)}]
set clock_port [get_ports $::env(PPA_CLOCK_PORT)]
set reset_port [get_ports $::env(PPA_RESET_PORT)]

create_clock -name core_clk -period $period $clock_port
set_clock_uncertainty -setup $uncertainty [get_clocks core_clk]

set input_filter "direction == input && name != $::env(PPA_CLOCK_PORT) && name != $::env(PPA_RESET_PORT)"
set data_inputs [get_ports * -filter $input_filter]
if {$::env(PPA_RESET_STYLE) eq "async"} {
    set_input_delay -clock core_clk -min 0.0 $reset_port
    set_input_delay -clock core_clk -max 0.0 $reset_port
    set_input_transition $::env(PPA_INPUT_TRANSITION_NS) $reset_port
    set_false_path -from $reset_port -to [get_pins -hierarchical */RESETN]
}

set_input_delay -clock core_clk -min $::env(PPA_INPUT_DELAY_MIN_NS) $data_inputs
set_input_delay -clock core_clk -max $io_delay $data_inputs
set_input_transition $::env(PPA_INPUT_TRANSITION_NS) $data_inputs

set outputs [get_ports * -filter {direction == output}]
set_output_delay -clock core_clk -min 0.0 $outputs
set_output_delay -clock core_clk -max $io_delay $outputs
set_load $::env(PPA_OUTPUT_LOAD_FF) $outputs
