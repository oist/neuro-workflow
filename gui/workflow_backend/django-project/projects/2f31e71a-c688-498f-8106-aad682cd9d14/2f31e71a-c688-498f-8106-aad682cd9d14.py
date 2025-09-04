def main():
    """Run a simple neural simulation workflow."""

    calc_1755561901166_6g0aiqz15 = SimulateSonataNetworkNode("SonataNetworkSimulation")
    calc_1755561901166_6g0aiqz15.configure(
        simulation_time=1000.0,
        record_from_population="internal",
        record_n_neurons=40
    )

    calc_1755561902414_ukmycsx9y = SimulateSonataNetworkNode("SonataNetworkSimulation")
    calc_1755561902414_ukmycsx9y.configure(
        simulation_time=1000.0,
        record_from_population="internal",
        record_n_neurons=40
    )

    calc_1755602423417_oyh5lfh8f = SimulateSonataNetworkNode("SonataNetworkSimulation")
    calc_1755602423417_oyh5lfh8f.configure(
        simulation_time=1000.0,
        record_from_population="internal",
        record_n_neurons=40
    )


    workflow = (
        WorkflowBuilder("neural_simulation")
            .add_node(calc_1755561901166_6g0aiqz15)
            .add_node(calc_1755561902414_ukmycsx9y)
            .add_node(calc_1755602423417_oyh5lfh8f)
            .build()
    )

    # Print workflow information
    print(workflow)
   
    # Execute workflow
    print("\nExecuting workflow...")
    success = workflow.execute()
   
    if success:
        print("Workflow execution completed successfully!")
    else:
        print("Workflow execution failed!")
        return 1
       
    return 0

if __name__ == "__main__":
    sys.exit(main())
