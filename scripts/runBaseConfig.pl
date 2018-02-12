#!/usr/bin/perl
#$ -S /usr/bin/perl

use List::Util 'shuffle';
use List::Util 'max';

# ***********************************************************************
# CHANGE THIS BEFORE EVERY SIMULATION!!!!
# ***********************************************************************
# WHAT COMPARISON IS THIS?

# WHERE TO STORE STATS FILES AND RESULTS

# Sam
# $stats = "/scratch/samgianelli/ml_cacheTuner/bbvs";
# $outputDir = "/scratch/samgianelli/ml_cacheTuner/data/results/energy";
# $bin_dir = "/scratch/benchmarks/SPEC2006_ARM";
# $simpoint = "/home/samjgianelli/CacheResearch/Scratch/Simulators/SimPoint.3.2/SimPoint.3.2/bin/";
# $arm = "/home/samjgianelli/CacheResearch/Scratch/Simulators/Gem5/gem5-stable/build/ARM";
# $configs = "/home/samjgianelli/CacheResearch/Scratch/Simulators/Gem5/gem5-stable/configs/spec2006";

# Diego 
$stats = "/home/diegojimenez/ECE523/stats";
$outputDir = "/home/diegojimenez/ECE523/outputDir";
$bin_dir = "/home/diegojimenez/ECE523/SPEC2006_ARM";
$simpoint = "/home/diegojimenez/ECE523/SimPoint.3.2/bin";
$arm = "/home/diegojimenez/ECE523/gem5/build/ARM";
$configs = "/home/diegojimenez/ECE523/gem5/configs/spec2006";

# ***********************************************************************
# END CHANGES
# ***********************************************************************

# HETEROGENEOUS BENCHMARKS - same as homogeneous combinations
# @benchmarks = ("leslie3d", "mcf", "astar", "bzip2", "gobmk", "libquantum", "milc", "namd", "xalancbmk", "omnetpp", "bwaves", "soplex", "hmmer", "h264ref", "calculix");
@benchmarks = ("leslie3d", "mcf", "astar", "bzip2", "libquantum", "namd", "xalancbmk", "omnetpp", "bwaves", "soplex", "hmmer", "h264ref", "calculix", "gcc");
# @benchmarks = ("leslie3d", "mcf", "astar");
# @benchmarks = ("leslie3d");
# $i = 14;

# my ($l1i_size, $l1d_size, $l1i_assoc, $l1d_assoc, $cacheline_size, $clock, $i) = @ARGV;
# my ($i) = @ARGV;

# $l2_size = 2048;
# $l2_assoc = 8;

# $fastForward = 300000000;
# $maxInsts = 1000000000000;
# $interval = 100000000;

$maxInsts = 1000000;
$large = 90000000;
$small = 10000000;
$fastForward = 300000000;

# $BANK_SIZE = 1024;

# Cache config
$cacheline_size=64;
$l1d_size="32kB";
$l1d_assoc=4;
$l1i_size="32kB";
$l1i_assoc=4;

for($i = 0; $i < @benchmarks; $i++) {
	$start = time;
	
	# NOTE: In config file, clock is in ticks. 1ns = 1000 ticks. Clock period = 1/freq
	$cacheLineSize = $cacheline_size;

	$clock0 = $clock/1000 . "GHz";
	$l1isize0 = $l1i_size/1024 . "kB";
	$l1iassoc0 = $l1i_assoc;
	$l1dsize0 = $l1d_size/1024 . "kB";
	$l1dassoc0 = $l1d_assoc;

	$l2size = $l2_size/1024 . "MB";
	$l2assoc = $l2_assoc;

	# CPU 1
	$clock1 = $clock0;
	$l1isize1 = $l1isize0;
	$l1iassoc1 = $l1iassoc0;
	$l1dsize1 = $l1dsize0;
	$l1dassoc1 = $l1dassoc0;

	# CREATE NAMES
	$l1i_name = $l1i_size/1024;
	$l1d_name = $l1d_size/1024;

	$l1i_cfg = $l1i_name . "k" . $l1i_assoc . "w" . $cacheline_size;
	$l1d_cfg = $l1d_name . "k" . $l1d_assoc . "w" . $cacheline_size;

	$fastForward_name = "FF" . $fastForward / 1000000 . "m";
	$maxInsts_name = "Max" . $maxInsts / 1000000000 . "b";
	$interval_name = "Int" . $interval / 1000000 . "m";

	$simp_cfg = $maxInsts_name;

	# create simulation directory
	if(!(-d "$stats/$benchmarks[$i]"))
	{
		system("/bin/mkdir $stats/$benchmarks[$i]");
	}
	# $sim_dir_name = "$stats/$benchmarks[$i]";
	$sim_dir_name = "$stats/$benchmarks[$i]/$simp_cfg";		#/$l1i_cfg-$l1d_cfg-$clock0";
	if(!(-d "$sim_dir_name"))
	{
		system("/bin/mkdir $sim_dir_name");
	}
	$wait = 1;
	$cache = "--caches --cacheline_size=$cacheline_size --l1d_size=$l1d_size --l1d_assoc=$l1d_assoc --l1i_size=$l1i_size --l1i_assoc=$l1i_assoc";

	print "\n================================= BENCHMARK $benchmarks[$i] =================================\n";
	# print "rm -r $sim_dir_name/*\n";
	system("rm -r $sim_dir_name/cpt.*\n");
	
	$statsFileName = "$sim_dir_name/$benchmarks[$i].txt";
	$firstRun = "$arm/gem5.fast --stats-file=$statsFileName --outdir=$sim_dir_name --dump-config=$sim_dir_name/config.ini $configs/spec2006_se_sg.py --checkpoint-at-end --cpu-type=AtomicSimpleCPU -n 1 --mem-size=8192MB --maxinsts=$fastForward --b $benchmarks[$i]";
	print "$firstRun\n\n";
	system("$firstRun");

	$previousCPT = "";
	for ($j = 1; $j <= 1000; $j = $j + 1){
		print "\n======================================== INTERVAL $j ========================================\n";
		$number = sprintf("%04d", $j);
		$statsFileName = "$sim_dir_name/$benchmarks[$i]$number.a.txt";
		$smallRun = "$arm/gem5.fast --stats-file=$statsFileName --outdir=$sim_dir_name --dump-config=$sim_dir_name/config.ini $configs/spec2006_se_sg.py --checkpoint-at-end --checkpoint-restore=1 $cache --cpu-type=AtomicSimpleCPU -n 1 --mem-size=8192MB --maxinsts=$small --b $benchmarks[$i]";
		print "$smallRun\n\n";
		system("$smallRun");

		if ($j == 1){
			if(!open(READ_FILE, "$sim_dir_name/$benchmarks[$i].txt")) {
				print "Cannot find $sim_dir_name/$benchmarks[$i].txt";
			}
			@in_lines = <READ_FILE>;
			foreach $key(@in_lines) {
				if($key =~ /final_tick/) {
					$key =~ s/final_tick//;
					@tmp = split(' ', $key);
					$finalTick = $tmp[0];
				}
			}
			print "Deleting first cpt file rm -r $sim_dir_name/cpt.$finalTick";
			system("rm -r $sim_dir_name/cpt.$finalTick")
		}
		

		if(!open(READ_FILE, $statsFileName)) {
			print "Cannot find $statsFileName";
		}
		@in_lines = <READ_FILE>;
		foreach $key(@in_lines) {
			if($key =~ /final_tick/) {
				$key =~ s/final_tick//;
				@tmp = split(' ', $key);
				$finalTick = $tmp[0];
			}
		}
		if(!open(READ_FILE, $statsFileName)) {
			print "Cannot find $statsFileName";
		}
		@in_lines = <READ_FILE>;
		foreach $key(@in_lines) {
			if($key =~ /sim_ticks/) {
				$key =~ s/sim_ticks//;
				@tmp = split(' ', $key);
				$ticksRan = $tmp[0];
			}
		}
		$previousCPT = $finalTick - $ticksRan;
		$previousCPT = "$sim_dir_name/cpt.$previousCPT";
		print "\n\nDeleting: $previousCPT\n";
		system("rm -r $previousCPT");

		$statsFileName = "$sim_dir_name/$benchmarks[$i]$number.b.txt";
		$largeRun = "$arm/gem5.fast --stats-file=$statsFileName --outdir=$sim_dir_name --dump-config=$sim_dir_name/config.ini $configs/spec2006_se_sg.py --checkpoint-at-end --checkpoint-restore=1 $cache --cpu-type=AtomicSimpleCPU -n 1 --mem-size=8192MB --maxinsts=$large --b $benchmarks[$i]";
		print "$largeRun\n\n";
		system("$largeRun");

		print "Stats file: $statsFileName";
		if(!open(READ_FILE, $statsFileName)) {
			print "Cannot find $statsFileName";
		}
		@in_lines = <READ_FILE>;
		foreach $key(@in_lines) {
			if($key =~ /final_tick/) {
				$key =~ s/final_tick//;
				@tmp = split(' ', $key);
				$finalTick = $tmp[0];
			}
		}
		if(!open(READ_FILE, $statsFileName)) {
			print "Cannot find $statsFileName";
		}
		@in_lines = <READ_FILE>;
		foreach $key(@in_lines) {
			if($key =~ /sim_ticks/) {
				$key =~ s/sim_ticks//;
				@tmp = split(' ', $key);
				$ticksRan = $tmp[0];
			}
		}
		$previousCPT = $finalTick - $ticksRan;
		$previousCPT = "$sim_dir_name/cpt.$previousCPT";
		print "\n\nDeleting: $previousCPT\n";
		system("rm -r $previousCPT");
    }

	$duration = time - $start;
	print "Execution time: $duration s\n";	
	system("echo \"Done with: $benchmarks[$i]. It took $duration s.\" | mailx -s \"Run Done\" diegojimenez\@email.arizona.edu");
	system("echo \"Execution time: $duration s\n\" > $sim_dir_name/executionTime.txt");
}

system("echo \"Done with all runs!\" | mailx -s \"Run Done\" diegojimenez\@email.arizona.edu");

# Working 
# /home/diegojimenez/ECE523/gem5/build/ARM/gem5.fast --stats-file=/home/diegojimenez/ECE523/stats2/leslie3d/Max0.0001b/leslie3d.txt --outdir=/home/diegojimenez/ECE523/stats2/leslie3d/Max0.0001b --dump-config=/home/diegojimenez/ECE523/stats2/leslie3d/Max0.0001b/config.ini /home/diegojimenez/ECE523/gem5/configs/spec2006/spec2006_se_sg.py --checkpoint-at-end --cpu-type=AtomicSimpleCPU -n 1 --mem-size=8192MB --maxinsts=100000 --b leslie3d

# /home/diegojimenez/ECE523/gem5/build/ARM/gem5.fast --stats-file=/home/diegojimenez/ECE523/stats2/leslie3d/Max0.0001b/leslie3d0001.txt --outdir=/home/diegojimenez/ECE523/stats2/leslie3d/Max0.0001b --dump-config=/home/diegojimenez/ECE523/stats2/leslie3d/Max0.0001b/config.ini /home/diegojimenez/ECE523/gem5/configs/spec2006/spec2006_se_sg.py --checkpoint-at-end --checkpoint-restore=1 --caches --cacheline_size=64 --l1d_size=32kB --l1d_assoc=4 --l1i_size=32kB --l1i_assoc=4 --cpu-type=AtomicSimpleCPU -n 1 --mem-size=8192MB --maxinsts=100001 --b leslie3d