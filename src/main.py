from orchestrator.orchestrator import Orchestrator

# Secretary system main contact interface
# Can change this up to a proper interface in the future

def main():

    orchestrator = Orchestrator()

    print("Welcome to the Agent-agent system!")  # Change this into something else in the future, maybe.

    while True:

        input_text = input(">>> ")
        
        if input_text.lower() == "bye":

            # Loop until the user gives a valid y/n answer, rather than
            # silently exiting or saving on unrecognized input.
            while True:
                choice = input("Save this chat's memory? (y/n): ").strip().lower()

                if choice == "y":

                    saved_path = orchestrator.end_session(save=True)
                    print(f"Chat memory saved to {saved_path}.")
                    break

                elif choice == "n":

                    orchestrator.end_session(save=False)
                    print("Chat memory discarded.")
                    break

                else:
                    print("Invalid input. Please enter 'y' or 'n'.")

            break

        print(orchestrator.run_orchestrator(input_text))

if __name__ == "__main__":
    main()

