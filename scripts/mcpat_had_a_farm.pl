
use List::Util 'shuffle';
use List::Util 'max';

@benchmarks = ("xalancbmk", "namd", "mcf", "libquantum", "hmmer", "calculix", "bzip2", "bwaves", "leslie3d", "h264ref", "gcc", "gamess", "astar", "milc");

$stats = "/home/diegojimenez/ECE523/stats/processed_output";
$mcpatExecutable = "/home/diegojimenez/ECE523/mcpat/mcpat";
$mcpatScript = "/home/diegojimenez/ECE523/mcpat/m5-mcpat-3.pl";
$mcpatTemplate = "/home/diegojimenez/ECE523/mcpat/detailed_2-mcpat-template.xml";
$output = "/home/diegojimenez/ECE523/outputDir";

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


system("chmod 700 $mcpatExecutable $mcpatScript $mcpatTemplate")
if(!(-d "$output")) {
 system("/bin/mkdir $output");
}

opendir my $stats, "/some/path" or die "Cannot open directory: $!";
my @files = readdir $stats;
closedir $stats;



for ($x = 0; $x < @files; $x++){
    print("$files[$x]")
}