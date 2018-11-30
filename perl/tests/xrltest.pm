use strict;
use warnings;

package xrltest;

sub almost_equal {
	my $actual = shift;
	my $expected = shift;
	my $threshold = 1E-6;
	if (@_) {
		$threshold = shift;
	}

	return abs($actual - $expected) < $threshold;
}


1;
