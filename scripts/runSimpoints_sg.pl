#!/usr/bin/perl
#$ -S /usr/bin/perl

use List::Util 'shuffle';
use List::Util 'max';

#***********************************************************************
#CHANGE THIS BEFORE EVERY SIMULATION!!!!
#***********************************************************************
#WHAT COMPARISON IS THIS?

#WHERE TO STORE STATS FILES AND RESULTS
$stats = "/scratch/samgianelli/ml_cacheTuner/bbvs";
$outputDir = "/scratch/samgianelli/ml_cacheTuner/data/results/energy";
$bin_dir = "/scratch/benchmarks/SPEC2006_ARM";
$simpoint = "/home/samjgianelli/CacheResearch/Scratch/Simulators/SimPoint.3.2/SimPoint.3.2/bin/";
#***********************************************************************
#END CHANGES
#***********************************************************************

#HETEROGENEOUS BENCHMARKS - same as homogeneous combinations
#@benchmarks = ("leslie3d", "mcf", "astar", "bzip2", "gobmk", "libquantum", "milc", "namd", "xalancbmk", "omnetpp", "bwaves", "soplex", "hmmer", "h264ref", "calculix");
@benchmarks = ("leslie3d", "mcf", "astar", "bzip2", "libquantum", "namd", "xalancbmk", "omnetpp", "bwaves", "soplex", "hmmer", "h264ref", "calculix", "gcc");

#$i = 14;

$arm = "/home/samjgianelli/CacheResearch/Scratch/Simulators/Gem5/gem5-stable/build/ARM";
$configs = "/home/samjgianelli/CacheResearch/Scratch/Simulators/Gem5/gem5-stable/configs/spec2006";

#my ($l1i_size, $l1d_size, $l1i_assoc, $l1d_assoc, $cacheline_size, $clock, $i) = @ARGV;
#my ($i) = @ARGV;

#$l2_size = 2048;
#$l2_assoc = 8;

$fastForward = 300000000;
$maxInsts = 1000000000000;
$interval = 100000000;

#$BANK_SIZE = 1024;


for($i = 0; $i < @benchmarks; $i++) {

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
	#$sim_dir_name = "$stats/$benchmarks[$i]";
	$sim_dir_name = "$stats/$benchmarks[$i]/$simp_cfg";		#/$l1i_cfg-$l1d_cfg-$clock0";
	if(!(-d "$sim_dir_name"))
	{
		system("/bin/mkdir $sim_dir_name");
	}
	$simpoint_dir = "$sim_dir_name/simpoints";
	if(!(-d "$simpoint_dir"))
	{	
		system("/bin/mkdir $simpoint_dir");# or die "Cannot create directory $!";
	}



$statsFileName = "$sim_dir_name/$benchmarks[$i].txt";

print "\n================ GENERATING BASIC BLOCK VECTORS FOR $benchmarks[$i] =================================\n";
#GENERATE SIMPOINTS
system("$arm/gem5.fast --stats-file=$statsFileName --outdir=$sim_dir_name --dump-config=$sim_dir_name/config.ini $configs/spec2006_se_sg.py --cpu-type=atomic --fastmem -n 1 --fast-forward=$fastForward  --mem-size=8192MB --maxinsts=$maxInsts --simpoint-profile --simpoint-interval=$interval --b $benchmarks[$i]");
#print "\n\n ONE \n\n";
#EXTRACT simpoint.bb.gz
system("/bin/gunzip $sim_dir_name/simpoint.bb.gz");
#print "\n\n TWO \n\n";
system("/bin/mv $sim_dir_name/simpoint.bb $simpoint_dir/$benchmarks[$i].bb");
#print "\n\n THREE \n\n";
print "\n================ GENERATING SIMPOINTS FOR $benchmarks[$i] =================================\n";
#CREATE SIMPOINTS
system("$simpoint/simpoint -loadFVFile $simpoint_dir/$benchmarks[$i].bb -maxK 20 -saveSimpoints $simpoint_dir/$benchmarks[$i].simpoints -saveSimpointWeights $simpoint_dir/$benchmarks[$i].weights -saveLabels $simpoint_dir/$benchmarks[$i].labels");
#print "\n\n FOUR \n\n";
}
