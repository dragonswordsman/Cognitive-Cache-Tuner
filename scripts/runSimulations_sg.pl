#!/usr/bin/perl
#$ -S /usr/bin/perl

use List::Util 'shuffle';
use List::Util 'max';


#$ENV{'LD_LIBRARY_PATH'} = '/home/tosiron/bin/lib:/home/tosiron/bin/lib64:$LD_LIBRARY_PATH';
#$ENV{'PATH'} = '/home/tosiron/bin/bin:$PATH';
#***********************************************************************
#CHANGE THIS BEFORE EVERY SIMULATION!!!!
#***********************************************************************
#WHAT COMPARISON IS THIS?
$iq = 64;

$sam = "/scratch/samjgianelli/test";
#WHERE TO STORE STATS FILES AND RESULTS
$stats = "/scratch/samgianelli/test/stats";
$outputDir = "/scratch/samgianelli/test/results";
$bin_dir = "/scratch/benchmarks/SPEC2006_ARM";
$simpoint = "/home/samjgianelli/CacheResearch/Scratch/Simulators/SimPoint.3.2/SimPoint.3.2/bin";    #what do I do with this?
$bbvs_dir = "/scratch/samgianelli/test/bbvs";

#***********************************************************************
#END CHANGES
#***********************************************************************

#HETEROGENEOUS BENCHMARKS - same as homogeneous combinations
@benchmarks = ("leslie3d", "mcf", "astar", "bzip2", "libquantum", "namd", "xalancbmk", "omnetpp", "bwaves", "soplex", "hmmer", "h264ref", "calculix", "gcc");


#$i = 3;

$arm = "/home/samjgianelli/CacheResearch/Scratch/Simulators/Gem5/gem5-stable/build/ARM";
$configs = "/home/samjgianelli/CacheResearch/Scratch/Simulators/Gem5/gem5-stable/configs/spec2006";

my ($l1i_size, $l1d_size, $l1i_assoc, $l1d_assoc, $cacheline_size, $clock, $i) = @ARGV;

#$l2_size = 2048;
#$l2_assoc = 8;

$fastForward = 300000000;
$maxInsts = 1000000000;
$interval = 1000000;


#$BANK_SIZE = 1024;

#for($i = 2; $i < 4; $i++) {

					#NOTE: In config file, clock is in ticks. 1ns = 1000 ticks. Clock period = 1/freq
					$cacheLineSize = $cacheline_size;
					
					$clock0 = $clock/1000 . "GHz";
					$l1isize0 = $l1i_size/1024 . "kB";
					$l1iassoc0 = $l1i_assoc;
					$l1dsize0 = $l1d_size/1024 . "kB";
					$l1dassoc0 = $l1d_assoc;
					
					$l2size = $l2_size/1024 . "MB";
					$l2assoc = $l2_assoc;
					

					#CPU 1
					$clock1 = $clock0;
					$l1isize1 = $l1isize0;
					$l1iassoc1 = $l1iassoc0;
					$l1dsize1 = $l1dsize0;
					$l1dassoc1 = $l1dassoc0;
					
				#CREATE NAMES
					$l1i_name = $l1i_size/1024;
					$l1d_name = $l1d_size/1024;
					
					$l1i_cfg = $l1i_name . "k" . $l1i_assoc . "w" . $cacheline_size;
					$l1d_cfg = $l1d_name . "k" . $l1d_assoc . "w" . $cacheline_size;
					
					$fastForward_name = "FF" . $fastForward / 1000000 . "m";
					$maxInsts_name = "Max" . $maxInsts / 1000000000 . "t";
					$interval_name = "Int" . $interval / 1000000 . "m";
					
					$simp_cfg = $fastForward_name . "-" . $maxInsts_name . "-" . $interval_name;
					
					#create simulation directory
					if(!(-d "$stats/$benchmarks[$i]"))
						{
							system("/bin/mkdir $stats/$benchmarks[$i]");
						}
					$config_name = "$l1i_cfg-$l1d_cfg-$clock0";
					$sim_dir_name = "$stats/$benchmarks[$i]/$l1i_cfg-$l1d_cfg-$clock0";
					if(!(-d "$sim_dir_name"))
					{
						system("/bin/mkdir $sim_dir_name");
					}
					


$statsFileName = "$sim_dir_name/$benchmarks[$i].txt";


#TAKE OUT SIMPOINTS FROM simpoint.simpoints
$simpointFile = "$bbvs_dir/$benchmarks[$i]/$simp_cfg/simpoints/$benchmarks[$i].simpoints";

@simpoints_lines = "";

if(!open(READ_SIMPOINTS, $simpointFile))
{
	die "Shit! Can't open $simpointFile";
}

@simpoint_lines = <READ_SIMPOINTS>;

$s = 0;
foreach(@simpoint_lines) {
	@tmp = split(' ', $_);
	$simpoints[$s] = $tmp[0];
	$s++;
}

#$fastforward_unit = 300000000;   # 1000000000
#RUN SIMULATIONS FOR SIMPOINTS
$no_of_simpoints = @simpoints;

$j = 0;
for($j = 0; $j < @simpoints; $j++) { 

print "\n================ RUNNING SIMULATIONS FOR SIMPOINT $benchmarks[$i]-$j OF $no_of_simpoints =================================\n";
$fastforward = $fastForward + (1000000 * $simpoints[$j]);
#$fastforward = 1000000;

$combination = "$benchmarks[$i]-$j.txt";

$statsFileName = "$sim_dir_name/$combination";
#SE MODE

#exit;

system("$arm/gem5.fast --stats-file=$statsFileName  --dump-config=$sim_dir_name/config.ini $configs/spec2006_se_sg.py --caches --l1i_size=$l1isize0 --l1i_assoc=$l1iassoc0 --l1d_size=$l1dsize0 --l1d_assoc=$l1dassoc0 --cacheline_size=$cacheLineSize --cpu-clock=$clock0 --mem-size=8192MB --cpu-type=arm_detailed -n 1 --fast-forward=$fastforward --maxinsts=$interval --b $benchmarks[$i]");
# SAM: decreased max instruction cnt to 1000000 from 100000000
#exit;

#system("$arm/gem5.fast --stats-file=$statsFileName  --dump-config=$sim_dir_name/config.ini $configs/spec2006_se.py --caches --l1i_size=$l1isize0 --l1i_assoc=$l1iassoc0 --l1d_size=$l1dsize0 --l1d_assoc=$l1dassoc0 --cacheline_size=$cacheLineSize --l2cache --l2_size=$l2size --l2_assoc=$l2assoc --cpu-clock=$clock0 --cpu-type=arm_detailed -n 1 --fast-forward=$fastforward --maxinsts=1000000 --b $benchmarks[$i]");


######################################################################################################
					##########START ENERGY CALCULATIONS
					
					###******************************************************
					#NEED TO WORK ON THE PARSING!!!
					###******************************************************
					
					#call script to carry out EDP calculation
					#do 'energy.pl';
					#$MCPATPARSER = "/home/tosiron/heterogeneous/scripts";
					$SCRIPTS = "/scratch/samgianelli/scripts";
					$MCPAT = "/scratch/benchmarks/misc/mcpat";
					$mcpat_output = "$sim_dir_name/mcpat_output-$j.txt";
					$mcpat_template = "detailed_2-mcpat-template.xml";  # 1_mcpat-template.xml
					$mcpat_script = "m5-mcpat-3.pl";
					$power_file = "power-$j.xml";

					#copy mcpat parser to simulation directory
					#system("cp $MCPATPARSER/mcpatparser.py $sim_dir_name");
					system("/bin/cp $SCRIPTS/$mcpat_script $sim_dir_name");
					system("/bin/cp $SCRIPTS/$mcpat_template $sim_dir_name");
					#system("cd $sim_dir_name");
					#system("chmod 777 $sim_dir_name/mcpatparser.py");
					system("/bin/chmod 777 $sim_dir_name/$mcpat_script");
					system("/bin/chmod 777 $sim_dir_name/$mcpat_template");

#*************************************************************************
#ENSURE THAT CONFIG.INI FILE HAS CORRECT CONFIGURATIONS!!!
#*************************************************************************

					#system("$sim_dir_name/mcpatparser.py -s $sim_dir_name/$combination -C $sim_dir_name/config.ini -p $sim_dir_name/power.xml -y $sim_dir_name/summary.xml");
					#generate power.xml and summary.xml files
					
					system("/bin/perl $sim_dir_name/$mcpat_script $sim_dir_name/$combination $sim_dir_name/config.ini $sim_dir_name/$mcpat_template > $sim_dir_name/$power_file");
					
					#####################################################################33
					#Get the total cycles from power.xml file
					$str_temp = "$sim_dir_name/$power_file";

					@in_lines = "";
						
						if(!open(READ_FILE, $str_temp)) 
						{
							die "Cannot find $str_temp";
						}
						
					@in_lines = <READ_FILE>;
					close READ_FILE;

					#copy mcpat to simulation directory
					system("/bin/cp $MCPAT $sim_dir_name");
					
					system("/bin/chmod 777 $sim_dir_name/mcpat");
					#run mcpat on power.xml file, store output in mcpat_output
					
					print "\n================ CALCULATING POWER FOR $benchmarks[$i]-$j OF $no_of_simpoints =================================\n";
					system("$sim_dir_name/mcpat -infile $sim_dir_name/$power_file -print_level 4 > $mcpat_output");


					#Get peak power value from McPAT output
					$mcpat_look = "Peak Power";
					$str_temp = "$sim_dir_name/mcpat_output-$j.txt";
					$str_temp_2 = "$sim_dir_name/$combination";

					@in_lines = "";
						
						if(!open(READ_FILE, $str_temp)) 
						{
							die "Cannot find $str_temp";
						}
						
					@in_lines = <READ_FILE>;
					close READ_FILE;

					foreach $key(@in_lines) {
									
							if($key =~ /^*Peak Power*/) {
								$key =~ s/Peak Power//;
								@tmp = split(' ', $key);
								$peakPower = $tmp[1]; 
								#print("$peakPower\n");
								
							}
							
							if($key =~ /^*Core clock Rate(MHz)*/) {
								$key =~ s/Core clock Rate(MHz)//;
								@tmp = split(' ', $key);
								$frequency = $tmp[3]; 
								#print("$peakPower\n");
								
							}
					}
					
					@in_lines_2 = "";
					if(!open(READ_FILE_2, $str_temp_2)) {
						die "Cannot find $str_temp_2";
					}
					
					@in_lines_2 = <READ_FILE_2>;
					close READ_FILE_2;
					
					#NUMBER OF CYCLES
					foreach $key(@in_lines_2) {
						if($key =~ /system.switch_cpus.numCycles/) {
							$key =~ s/system.switch_cpus.numCycles//;
							@tmp = split(' ', $key);
							$totalCycles = $tmp[0];
						}
					}
					
					#IPC
					foreach $key(@in_lines_2) {
						if($key =~ /system.switch_cpus.commit.committedInsts/) {
							$key =~ s/system.switch_cpus.commit.committedInsts//;
							@tmp = split(' ', $key);
							$instructions = $tmp[0];
						}
					}
					
					#DATA CACHE MISS RATE
					foreach $key(@in_lines_2) {
						if($key =~ /system.cpu.dcache.overall_miss_rate::switch_cpus.data/) {
							$key =~ s/system.cpu.dcache.overall_miss_rate::switch_cpus.data//;
							@tmp = split(' ', $key);
							$dCacheMR = $tmp[0];
						}
					}
					
					#CPI
					foreach $key(@in_lines_2) {
						if($key =~ /system.switch_cpus.cpi_total/) {
							$key =~ s/system.switch_cpus.cpi_total//;
							@tmp = split(' ', $key);
							$cpi = $tmp[0];
						}
					}
					
					#IPC
					foreach $key(@in_lines_2) {
						if($key =~ /system.switch_cpus.ipc_total/) {
							$key =~ s/system.switch_cpus.ipc_total//;
							@tmp = split(' ', $key);
							$ipc = $tmp[0];
						}
					}
					
					#L2 CACHE MISS RATE
					foreach $key(@in_lines_2) {
						if($key =~ /system.l2.overall_miss_rate::total/) {
							$key =~ s/system.l2.overall_miss_rate::total//;
							@tmp = split(' ', $key);
							$l2CacheMR = $tmp[0];
						}
					}
					
					#L2 CACHE MISSES
					foreach $key(@in_lines_2) {
						if($key =~ /system.l2.overall_misses::total/) {
							$key =~ s/system.l2.overall_misses::total//;
							@tmp = split(' ', $key);
							$l2CacheMisses = $tmp[0];
						}
					}

					################################################
					#Calculate EDP
					$running_time = (1/($frequency * 1000000))*$totalCycles;
					$energy = $peakPower * $running_time;
					#calculate energy delay product
					$edp = $peakPower * ($running_time**2);
					#$cpi = $totalCycles/$instructions;
					
					#print("$peakPower;    $totalCycles;     $edp\n");
					
					#$store = ("$benchmarks[$i]-$j	$simpoints[$j]	$peakPower	$totalCycles	$running_time	$energy	$edp	$cpi	$ipc	$dCacheMR	$l2CacheMR	$l2CacheMisses	\n");
					
					$store = ("$config_name	$benchmarks[$i]-$j	$simpoints[$j]	$peakPower	$totalCycles	$running_time	$energy	$edp	$cpi	$ipc	$dCacheMR	\n");
					
					
					#PRINT TO AN OUTPUT FILE
					$out_file = ">>"."$outputDir/$benchmarks[$i].txt"; 
					open(WRITE_FILE, $out_file);
					#print WRITE_FILE "$sim_dir_name";
					print WRITE_FILE "$store";
					#print WRITE_FILE "\n\n";
					close WRITE_FILE;
					
					$count++;
					
					################################################
					#system("rm $sim_dir_name/power.xml");
					#system("rm $sim_dir_name/summary.xml");
					system("/bin/rm $sim_dir_name/$mcpat_template");
					system("/bin/rm $sim_dir_name/$mcpat_script");
					system("/bin/rm $sim_dir_name/mcpat");
					#system("rm $sim_dir_name/config.ini");
					#system("rm $sim_dir_name/$benchmarks[$i].txt");
					#
					system("/bin/mv $sim_dir_name/$power_file $sim_dir_name/$benchmarks[$i]-$j.xml");


print "\n================ DONE RUNNING SIMULATIONS FOR $benchmarks[$i]-$j OF $no_of_simpoints =================================\n";
#}
#exit;

#Benchmark combination already exists, so get out:

}

exit;
