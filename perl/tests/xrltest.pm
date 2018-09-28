use strict;
use warnings;

package xrltest;

sub almost_equal {
	my $actual = shift;
	my $expected = shift;

	return abs($actual - $expected) < 1E-6;
}


1;
