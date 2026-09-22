import settings
import compare_main
import minlp_main

settings.parse_cli()

if settings.mode == "solve":
    minlp_main.run()
else:
    compare_main.run()
