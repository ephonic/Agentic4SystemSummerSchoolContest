proc require_env {name} {
    if {![info exists ::env($name)] || $::env($name) eq ""} {
        error "missing environment variable $name"
    }
    return $::env($name)
}

set top [require_env PPA_TOP]
set filelist [require_env PPA_FILELIST_ABS]
set map_lib [require_env PPA_MAP_LIB]
set comb_libs [require_env PPA_COMB_LIBS]
set seq_lib [require_env PPA_SEQ_LIB]
set netlist [require_env PPA_NETLIST]
set json [require_env PPA_NETLIST_JSON]
set stat_report [require_env PPA_SYNTH_STAT]
set abc_delay [require_env PPA_ABC_DELAY_PS]

foreach lib [split $comb_libs ":"] {
    yosys read_liberty -lib -ignore_miss_func $lib
}
yosys read_liberty -lib -ignore_miss_func $seq_lib

if {[info exists ::env(PPA_SRAM_MODELS)] && $::env(PPA_SRAM_MODELS) ne ""} {
    foreach model [split $::env(PPA_SRAM_MODELS) ":"] {
        yosys read_verilog -sv -lib $model
    }
}

set include_args {}
set sources {}
set fp [open $filelist r]
while {[gets $fp line] >= 0} {
    set line [string trim $line]
    if {$line eq "" || [string match "#*" $line]} {
        continue
    }
    if {[string match "+incdir+*" $line]} {
        lappend include_args -I[string range $line 8 end]
    } else {
        lappend sources $line
    }
}
close $fp

if {[llength $sources] == 0} {
    error "filelist contains no RTL sources: $filelist"
}

foreach source $sources {
    yosys read_verilog -sv -defer {*}$include_args $source
}

yosys hierarchy -check -top $top
yosys synth -top $top -noabc
yosys dfflibmap -liberty $seq_lib
yosys abc -fast -D $abc_delay -liberty $map_lib
yosys clean -purge
yosys check -assert
yosys tee -o $stat_report stat
yosys write_verilog -noattr -noexpr -nodec $netlist
yosys write_json $json
