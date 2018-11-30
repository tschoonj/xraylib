<?

// SWIG exception codes
const ValueError = -9;

set_error_handler(function ($errno, $errstr, $errfile, $errline, $errcontext) {
	printf("\nError detected: %s\n", $errstr);
	exit(1);
});

set_exception_handler(function ($exception) {
	printf("\nMessage: %s\n", $exception->getMessage());
	printf("Traceback: %s\n", $exception->getTraceAsString());
	exit(1);
});


function assertAlmostEqual($actual, $expected, $threshold = 1.0E-6) {
	if (is_array($actual) && is_array($expected)) {
		if (count($actual) != count($expected)) {
			throw new Exception("assertAlmostEqual: actual and expected have different array lengths!");
		}
		$lambda = function($a, $e) use ($threshold) {
			if (abs($a - $e) > $threshold) {
				throw new Exception("assertAlmostEqual: $a and $e differ by too much!");
			}
			return TRUE;
		};
		array_map($lambda, $actual, $expected);
	}
	else if (abs($actual - $expected) > $threshold) {
		throw new Exception("assertAlmostEqual: actual and expected differ too much");
	}
}

function assertTrue($value) {
	if ($value != TRUE) {
		throw new Exception("assertTrue: assertion failed");
	}
}

function assertEqual($actual, $expected) {
	if ($actual != $expected) {
		throw new Exception("assertEqual: actual and expected are not equal!");
	}
}

function assertException($code, $function, ...$args) {
	try {
		call_user_func_array($function, $args);
	} catch (Exception $exception) {
		assertEqual($exception->getCode(), $code);
		return;
	}
	throw new Exception("assertException: no exception was thrown!");
}

// taken from http://php.net/manual/en/function.get-class-methods.php#118330
function get_bar($text) {
	$bar = "";
	for($i = 1 ; $i <= strlen($text) ; $i++) {
		$bar .= "=";
	}
	return $bar;
}

class XrlTestSuite {
	function __construct() {
		$this->test_array = array();
	}

	function append($test) {
		if (!is_a($test, "XrlTest")) {
			throw new Exception("XrlTestSuite::append test must be instance of XrlTest");
		}
		array_push($this->test_array, $test);
	}

	function run() {
		$failed = 0;
		foreach ($this->test_array as $test) {
			$failed += $test->run_tests();
		}
		exit($failed == 0 ? 0 : 1);
	}
}

class XrlTest {
	function __construct() {
		//$this->run_tests();
	}
	// run the tests
	function run_tests() {
		$class = get_class($this);
		$test_methods = preg_grep('/^test/', get_class_methods($this));
		$failed = 0;
		foreach ($test_methods as $method) {
			$start_rep = "test: $class::$method";
			$bar = get_bar($start_rep);
			print("\n$start_rep\n$bar\n");
			try {
				$this->$method();
				print("OK\n");
			} catch (Exception $exception) {
				printf("FAIL\nTraceback:\n%s\n", $exception->getTraceAsString());
				$failed++;
			}
		}
		return $failed;
    	}
}



?>
