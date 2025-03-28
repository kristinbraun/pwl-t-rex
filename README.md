# PWL-T-Rex
## Piecewise Linear Tool for Relaxations

```text                                                                        
                       @@   @@                                            
                    @           @                                         
                  @@@@@          @                                        
                @          @@     @                                       
              @      @@        @@ @                                       
            @             @@    @@@@                                      
        @@@       @@@@        @@@   @                                     
     @            @  @@         @   @                                     
   @              @@@@           @                                        
  @   @@                         @                                        
  @                              @                                        
  @                 @                                                     
   @     @@  @   @          @    @                                        
    @ @@                           @@                                     
     @                    @                @@@                            
       @     =@@@@@@@@@                         @@                        
                      @ @@                         @                      
                       @ @@             PWL         @                     
                        @               -T-           +                   
                       -@ @             Rex           @                   
                   @@@     @     @                     @                  
                  @@@  @@@@      @      *               @          @+     
                    @  @@      @@      @                  @        @@     
                      @@  @@@     @    @            @       @@@ :@@       
                     @:@@:  @@       @@             @               @     
                               @                   @               @@     
                              @  @     @           @@@           @@@      
                                 @  @@  @           @      @@@   @        
                                   @      @         @@@@@  +@@@           
                                            @@      @                     
                                @        @           @                    
                                        @    @       @                    
                              @@@@@@@@@    @        @                     
                                    @     @@       @                      
                                         @@@@@@@@@*                       
                                               -                          
```

This project implements and compares different Mixed Integer Programming (MIP) representation methods for reformulating Mixed-Integer Nonlinear Programs.

## Features

- Multiple MIP formulation methods:
  - Disaggregated Convex Combination
  - Logarithmic Disaggregated Convex Combination
  - Aggregated Convex Combination
  - Logarithmic Aggregated Convex Combination
  - Incremental Method
  - Multiple Choice Model
  - Binary Zig-Zag Model
  - Integer Zig-Zag Model

## Installation

1. Clone this repository:
```bash
git clone https://github.com/kristinbraun/pwl-t-rex.git
```

2. Install the required dependencies:
```bash
pip install pyomo gurobipy numpy scipy beautifulsoup4 lxml
```

### Dependencies

Core dependencies:
- pyomo - Mathematical optimization modeling
- gurobipy - Gurobi optimization solver interface
- numpy - Numerical computing
- scipy - Scientific computing utilities
- beautifulsoup4 - XML/HTML parsing
- lxml - XML/HTML processing library


## Usage

Run the main script with your desired parameters:

```bash
python pwltrex.py [filename] [options]
```

### Command Line Arguments

- `filename`: Path to the input file (required)
- `--method`: MIP method selection (-5 to 8)
  - -5: Run all MIP reformulations
  - -1: Initial MINLP
  - 0: 1D-MINLP
  - 1: DisaggConvex
  - 2: LogDisaggConvex
  - 3: AggConvex
  - 4: LogAggConvex
  - 5: Incremental
  - 6: MultipleChoice
  - 7: BinaryZigZag
  - 8: IntegerZigZag
- `--epsilon`: Error tolerance (default: 1)
- `--relax`: Relaxation method (0: Approximation, 1: Exact error, 2: Fixed error)
- `--timelimit`: Time limit for MILPs in seconds (default: 60)
- `--create`: Create model without solving (0: No, 1: Yes)
- `--solver_output`: Print solver output (0: No, 1: Yes)

### Example

```bash
python pwltrex.py alkyl.osil --method -5 --epsilon 0.1 --timelimit 120
```

## Output

The program outputs:
- Objective function value
- Runtime to optimal and to first primal solution
- Sorted rankings of methods by:
  - Total runtime
  - Time to first primal solution

## Project Structure
- `pwltrex.py`: Main execution script
- `MIPRef_osilToOnedim.py`: Converts OSIL format to one-dimensional representation 
- `MIPRef_onedimToMIP.py`: Converts one-dimensional to MIP representation
- `MIPRef_mipRepresentations.py`: Contains MIP formulation methods
- `MIPRef_graycode.py`: Gray code utilities for binary encoding
- `MIPRef_linrelax.py`: Linear relaxation utilities
- `nltree.py`: Nonlinear expression tree implementation
- `evaluation_solving.py`: Solving utilities
- `evaluation_statistics.py`: Statistical analysis utilities


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PWL-T-Rex in your research, please cite:

Braun, K., & Burlacu, R. (2023). A Computational Study for Piecewise Linear Relaxations of Mixed-Integer Nonlinear Programs. *Optimization Online*. Retrieved from https://optimization-online.org/2023/09/a-computational-study-for-piecewise-linear-relaxations-of-mixed-integer-nonlinear-programs/

BibTeX:
```bibtex
@article{braun2023computational,
    title={A Computational Study for Piecewise Linear Relaxations of Mixed-Integer Nonlinear Programs},
    author={Braun, Kristin and Burlacu, Robert},
    journal={Optimization Online},
    year={2023},
    url={https://optimization-online.org/2023/09/a-computational-study-for-piecewise-linear-relaxations-of-mixed-integer-nonlinear-programs/}
}
```

