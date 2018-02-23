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
$arm = "/home/diegojimenez/ECE523/gem5/build/ARM";
$configs = "/home/diegojimenez/ECE523/gem5/configs/spec2006";
$labels = "/home/diegojimenez/ECE523/data/100b_simpoints/simpoints";

# ***********************************************************************
# END CHANGES
# ***********************************************************************

# HETEROGENEOUS BENCHMARKS - same as homogeneous combinations
@benchmarks = ("xalancbmk", "namd", "mcf", "libquantum", "hmmer", "calculix", "bzip2", "bwaves", "leslie3d", "h264ref", "gcc", "gamess", "astar", "milc");
# @benchmarks = ("mcf");

# my ($l1i_size, $l1d_size, $l1i_assoc, $l1d_assoc, $cacheline_size, $clock, $i) = @ARGV;
# my ($i) = @ARGV


$maxInsts = 9000000;
$fastForward = 100000000;

# $BANK_SIZE = 1024;

# Simpoint Interval Length
$interval = 10000000;

# Default Cache config
$cacheline_size=64;
$l1d_size="32kB";
$l1d_assoc=4;
$l1i_size="32kB";
$l1i_assoc=4;

# Exploration Space
@cacheSize = ("2", "4", "8", "16", "32");
@associativity = ("1", "2", "4");
@lineSize = ("16", "32", "64");

for ($p = 0; $p < 126; $p++){
	$currBench = int($p / 9);
	$line = int(int($p / 3) % 3);
	$assoc_d = int($p % 3);
	# if (int($p % 13))
	# print "$p\t$currBench $line $assoc_d \n";	
	print "\n================================= BENCHMARK $benchmarks[$currBench] =================================\n";

	$checkpoint_dir = "$stats/$benchmarks[$currBench]";
	if(!(-d "$checkpoint_dir")) {
		system("/bin/mkdir $checkpoint_dir");
	}

	$labelBench = "$labels/$benchmarks[$currBench].simpoints";
	# print "$labelBench\n";
	open(READ_SIMPOINTS, $labelBench) or die "Could not open file '$labelBench' $!";
	@simpoint_lines = <READ_SIMPOINTS>;
	for ($j = 0; $j < @simpoint_lines; $j++){
		@temp = split(' ', $simpoint_lines[$j]);
		$simpoint = $temp[0];
		$startInst = $simpoint * $interval;
		$sim_dir_name = "$stats/$benchmarks[$currBench]/simpoint_$simpoint";
		$checkpoint_dir = $sim_dir_name;
		if(!(-d "$sim_dir_name")) {
			system("/bin/mkdir $sim_dir_name");
		}

		print "\n================================= USING SIMPOINT $startInst =================================\n";

		for ($size_d = 0; $size_d < @cacheSize; $size_d++) {
			for ($size_i = 0; $size_i < @cacheSize; $size_i++) {
				for ($assoc_i = 0; $assoc_i < @associativity; $assoc_i++) {
					if ($cacheSize[$size_d] / 2 < $associativity[$assoc_d] || $cacheSize[$size_i] / 2 < $associativity[$assoc_i]){
						next;
					}
					$l1i_cfg = $cacheSize[$size_i] . "kB" . $associativity[$assoc_i] . "w" . $lineSize[$line];
					$l1d_cfg = $cacheSize[$size_d] . "kB" . $associativity[$assoc_d] . "w" . $lineSize[$line];
					$sim_dir_name = "$stats/$benchmarks[$currBench]/simpoint_$simpoint/$l1i_cfg-$l1d_cfg";
					if(!(-d "$sim_dir_name")) {
						# print "$sim_dir_name\n";
						system("/bin/mkdir $sim_dir_name");
					}

					$cache = "--caches --cacheline_size=$lineSize[$line] --l1d_size=$cacheSize[$size_d]kB --l1d_assoc=$associativity[$assoc_d] --l1i_size=$cacheSize[$size_i]kB --l1i_assoc=$associativity[$assoc_i]";
					# print "$cache\n";
					$statsFileName = "$sim_dir_name/$benchmarks[$currBench].txt";
					$run = "$arm/gem5.fast --stats-file=$statsFileName --outdir=$sim_dir_name --dump-config=$sim_dir_name/config.ini $configs/spec2006_se_sg.py --checkpoint-restore=1 --checkpoint-dir=$checkpoint_dir $cache --cpu-type=AtomicSimpleCPU -n 1 --mem-size=8192MB --maxinsts=$maxInsts --b $benchmarks[$currBench]\n\n";
					print "$run\n";
					system("$run");
				}	
			}	
		}
		# $run = "rm -r $checkpoint_dir/cpt.*";
		# print "$run\n";
		# system($run);
	}
	close READ_SIMPOINTS;
}
