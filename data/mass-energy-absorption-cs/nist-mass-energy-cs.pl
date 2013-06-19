#!/usr/bin/env perl
#
#
#A little script that fetches the individual per-element tables of the mass energy-absorption coefficients from nist.gov
#and modifies them for easy introduction in xraylib

use strict;
use warnings;
use WWW::Curl::Easy;
use xraylib;

my $xcom_url = "http://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab";

my $curl = WWW::Curl::Easy->new;

open (OUTPUT, ">", "cs_energy_perl.txt");
printf OUTPUT "92\n";
for (my $i = 1 ; $i <= 92 ; $i++) {

	my $url = sprintf("%s/z%02i.html", $xcom_url, $i);
	print "URL: ".$url."\n";

	#use curl to download the files
	$curl->setopt(CURLOPT_URL, $url);
	my $html_file="";
	open(my $fileb, ">", \$html_file);

	$curl->setopt(CURLOPT_WRITEDATA, \$fileb);
	my $rv = $curl->perform;
	if ($rv != 0) {
		printf "Curl error for file %s: %s\n", $xcom_url, $curl->getinfo(CURLINFO_HTTP_CODE);
		exit 1;
	}

	#if success -> parse output for the ascii table
	if ($html_file =~ m/\)\n_{15,}\n ?\n{1,2}(.+) ?\n?<\/PRE/s) {
		print "match\n"	
	}
	else {
		print "no match\n";
		exit 1;
	}

	#print $1 if ($i == 1);
	#separate based on newlines
	my @lines = split /\n/,$1;	
	my @energies;
	my @cs;
	my @cs_en;
	foreach my $line (@lines) {
		$line=~ s/^\s+//;
		$line =~ s/\s+$//;
		my @vals = split /\s+/,$line;
		if (scalar(@vals) == 4) {
			$energies[-1] -= 0.00001;
			push @energies, $vals[1]*1E3;
			push @cs, $vals[2];
			push @cs_en, $vals[3];
		}
		else {
			push @energies, $vals[0]*1E3;
			push @cs, $vals[1];
			push @cs_en, $vals[2];
		}
	}
	printf OUTPUT "%i\n", scalar(@energies);
	for (my $j = 0 ; $j < scalar(@energies) ; $j++) {
		#take into account kissel cross sections
		my $ratio = &xraylib::CS_Total_Kissel($i,$energies[$j])/$cs[$j] ;
		$cs_en[$j] *= $ratio;
		printf OUTPUT "%f     %f\n", $energies[$j], $cs_en[$j];
	}


	close($fileb);

} 
close OUTPUT;




