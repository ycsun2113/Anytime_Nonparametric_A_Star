# ANA*: Anytime Nonparametric A*

Final project for the [ROB 422: Introduction to Algorithmic Robotics](https://berenson.robotics.umich.edu/courses/fall2024iar/index.html) course at the University of Michigan.

This project implements the A* and [ANA*: Anytime Nonparametric A*](https://ojs.aaai.org/index.php/AAAI/article/view/7819) algorithms to solve 2D navigation tasks for a PR2 robot in PyBullet simulation environments.

ANA* is one of the variants of A* algorithm that can find the suboptimal solution in each iteration to gradually find the optimal solution for path planning problem and can find the solution faster than the A* algorithm. The comparison of the solution cost vs. time for A* and ANA* was shown in Section 3 in the [report](./doc/report.pdf). Additionally, an admissible heuristic function is crucial for A* and its variants to find the optimal path. Therefore, several choices of the heuristic function will also be compared in this project.
