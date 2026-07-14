proc require_env {name} {
    if {![info exists ::env($name)] || $::env($name) eq ""} {
        error "missing environment variable $name"
    }
    return $::env($name)
}

foreach lib [split [require_env PPA_STA_LIBS] ":"] {
    read_liberty $lib
}

read_verilog [require_env PPA_NETLIST]
link_design [require_env PPA_TOP]
source [require_env PPA_SDC]

report_units
puts "PPA_CHECK_SETUP_BEGIN"
check_setup -verbose
puts "PPA_CHECK_SETUP_END"
report_checks -path_delay max -format full_clock_expanded -fields {slew cap input_pin net fanout} -digits 6
report_checks -path_delay min -format full_clock_expanded -fields {slew cap input_pin net fanout} -digits 6
report_worst_slack -max -digits 6
report_worst_slack -min -digits 6
report_tns -max -digits 6
report_tns -min -digits 6
report_clock_min_period
