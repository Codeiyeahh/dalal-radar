import json
from tools.registry import registry

def tool(name, description, parameters, toolset="default"):
    """
    A custom @tool decorator that registers the function with the Hermes Agent registry.
    This fulfills the '@tool decorator' request while staying natively compatible
    with NousResearch/hermes-agent's registry system.
    """
    def decorator(func):
        schema = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        
        def handler(args, **kwargs):
            try:
                # Remove Hermes specific kwargs that might not be in the function signature
                clean_args = {k: v for k, v in args.items()}
                res = func(**clean_args)
                # Hermes handlers MUST return a JSON string
                if isinstance(res, str):
                    # Check if it's already a JSON string
                    try:
                        json.loads(res)
                        return res
                    except json.JSONDecodeError:
                        return json.dumps({"result": res}, ensure_ascii=False)
                return json.dumps(res, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)

        registry.register(
            name=name,
            toolset=toolset,
            schema=schema,
            handler=handler,
            check_fn=lambda: True,
        )
        return func
    return decorator
