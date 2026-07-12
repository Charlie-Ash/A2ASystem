from orchestrator.orchestrator import Orchestrator

# Secretary system main contact interface
# Can change this up to a proper interface in the future

def main():

    orchestrator = Orchestrator()

    print("Welcome to the Agent-agent system!")  # Change this into something else in the future, maybe.

    while True:

        input_text = input(">>> ")
        if input_text.lower() == "bye":
            break

        print(orchestrator.run_orchestrator(input_text))

if __name__ == "__main__":
    main()

