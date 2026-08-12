import asyncio

from orchestrator.orchestrator import Orchestrator

# Secretary system main contact interface
# Can change this up to a proper interface in the future

async def main():

    # Orchestrator.create() discovers remote A2A agents over the network
    # (see tool_router.py/remote_agent.py), which is why construction is now
    # async instead of a plain Orchestrator() call.
    orchestrator = await Orchestrator.create()

    print("Welcome to the Agent-agent system!")  # Change this into something else in the future, maybe.

    while True:

        # input() itself is still a blocking call -- fine here, since
        # nothing else needs to run concurrently while waiting on it.
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

        print(await orchestrator.run_orchestrator(input_text))

if __name__ == "__main__":
    asyncio.run(main())
