import os, argparse, pandas as pd
from load_model import load_model
root_dir = os.path.dirname(os.path.abspath(__file__))

from helper import save_json, set_seed, get_result_path
from load_dataset import load_dataset, get_prompt
from environment import TRANSFORMER_CACHE
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMER_CACHE
from configs import task_dataframe, supported_models, prompt_type_options

def infer_model(
    call_model,
    prompt_type,
    text_inputs,
    image_inputs,
    task_id,
    model,
    gen_mode,
    low_semantic = False,
):  
    if prompt_type == 'cot':
        query = get_prompt(
            text_inputs,
            image_inputs,
            prompt_type,
            task_id, 
            model,
            'general', 
        )
        out = call_model(query)
        print('-------------------')
        print("CoT step:")
        print(f"{out['description']}\n")
        
        query = get_prompt(
            [],
            [],
            prompt_type,
            task_id, 
            model,
            gen_mode, 
            history = out['history'],
        )
        query['instruction'] = [query['instruction'][0], query['instruction'][1] + f"'{text_inputs[-1]}'."]
        print(f"Question: {query['instruction'][1]}")
        out = call_model(query)
    ############### Low Semantic ###############
    elif low_semantic:
        query = get_prompt(
            text_inputs,
            image_inputs,
            prompt_type,
            task_id, 
            model,
            gen_mode, 
        )
        print("========== query ===========")
        context = f"We provide a sequence of examples. Each example consists of a color and an object, paired with an image of the described object. Please notice that we are using nonsense strings to replace natural language. Please infer pattern from it. The names for these few examples are {text_inputs[0:4]}. "
        question = f"Based on these examples, you are required to generate a new image. The new image must strictly match the requested color and object, as described in the final input. You need to generate the image that matches {text_inputs[-1]}"
        query['instruction'] = [context, question]
        print(query)
        out = call_model(query)
    ############### Default ###############
    else:
        query = get_prompt(
            text_inputs,
            image_inputs,
            prompt_type,
            task_id, 
            model,
            gen_mode, 
        )
        out = call_model(query)
    return out

def inference(
    model,
    call_model,
    shot,
    prompt_type,
    task_id,
    overwrite,
    gen_mode,
    finetuned_model = False,
    data_mode = 'default', # ['default', 'ft_test']
    ft_mode = 'all',
    eval_task_theme = '',
    low_semantic = False,
):
    if finetuned_model and data_mode != 'ft_test':
        raise ValueError(f"finetuned models only supports loading ft_test data. You are considering {data_mode} data.")
        
    if (ft_mode == 'leave_one_out' and (not eval_task_theme)) or (ft_mode == 'all' and eval_task_theme):
        raise ValueError(f"ft_mode and eval_task_theme are incompatible!")
    
    if (ft_mode == 'leave_one_out'):
        if task_dataframe[task_id]['task_name'].split('-')[0].lower() != eval_task_theme:
            return None
    
    if low_semantic:
        true_shot = shot * 2
        base_path = get_result_path(
            finetuned_model,
            data_mode,
            model,
            gen_mode,
            true_shot,
            prompt_type,
            ft_mode,
            eval_task_theme,
        )
    else:
        base_path = get_result_path(
            finetuned_model,
            data_mode,
            model,
            gen_mode,
            shot,
            prompt_type,
            ft_mode,
            eval_task_theme,
        )
    
    folder_path = f"{base_path}/task_{task_id}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    data_loader = load_dataset(
        shot,
        prompt_type,
        task_id,
        data_mode = data_mode,
        ft_mode = ft_mode,
        low_semantic = low_semantic,
    )
    
    for count in range(len(data_loader)):

        input_dict = data_loader[count]
        text_inputs, image_inputs = input_dict["text_inputs"], input_dict["image_inputs"]
        save_path = f"{folder_path}/{input_dict['save_path']}"
        
        print(f"===={count}-th sample====")
        print(f"theta: {input_dict['theta']}")
        # for i in range(shot+1):
        #     print(f"{text_inputs[i]}")
        print("text_inputs: ")
        print(f"{text_inputs}")

        # skip if file exists
        if gen_mode == 'text':
            if not overwrite and os.path.exists(save_path+'.json'):
                print('skip')
                continue
        elif gen_mode == 'image':
            if not overwrite and os.path.exists(save_path+'.jpg'):
                print('skip')
                continue
        else:
            raise NotImplementedError(f"Unknown gen_mode: {gen_mode}!")
        
        out = infer_model(
            call_model,
            prompt_type,
            text_inputs,
            image_inputs,
            task_id,
            model,
            gen_mode,
            low_semantic,
        )
            
        out['text_inputs'] = text_inputs
        out['image_inputs'] = image_inputs
        if gen_mode == 'text':
            save_json(out, save_path+'.json')
        elif gen_mode == 'image':
            img = out['image']
            if img != None: img.save(save_path+'.jpg')
            out.pop('image')
            save_json(out, save_path+'.json')
        print("discription of out")
        print(f"{out['description']} \n")

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Generate images or image descriptions')
    parser.add_argument('--shot', type=int, nargs='+', default=[2,4,6,8, 12, 16])
    parser.add_argument('--prompt_type', type=str, nargs='+', default=['default'], choices=prompt_type_options)
    parser.add_argument('--model', type=str, default="qwen", choices = supported_models)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', nargs='+', type=str, default=['cuda']) # or ['35GiB', '25GiB', '35GiB']
    parser.add_argument('--task_id', type=int, nargs='+', default=list(task_dataframe.keys()))
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1])
    parser.add_argument('--gen_mode', type=str, default="image", choices=['text', 'image'])
    parser.add_argument('--finetuned_model', type=int, default=0, choices=[0,1], help = "whether to use finetuned model")
    parser.add_argument('--data_mode', type=str, default="default", choices=['default', 'ft_test'], help = "what dataset to use")
    parser.add_argument('--api_key', type=str, default="yz", help = "which key to use")
    parser.add_argument('--ft_mode', type=str, default='all', choices = ['all', 'leave_one_out'], help='finetune mode')
    parser.add_argument('--eval_task_theme', type = str, default = '', choices = ['', 'color', 'background', 'style', 'action', 'texture'], help = 'task theme for evaluation')
    parser.add_argument('--low_semantic', type=int, default=0, choices=[0,1], help = "choose if exclude natural language")

    args = parser.parse_args()
    
    # print experiment configuration
    args_dict = vars(args)
    print("########"*3)
    print('## Experiment Setting:')
    print("########"*3)
    for key, value in args_dict.items():
        print(f"| {key}: {value}")
    
    if len(args.device) == 1: 
        device = args.device[0]
    else:
        device = {}
        for i in range(len(args.device)):
            device[i] = args.device[i]

    set_seed(args.seed)
    
    if args.finetuned_model:
        if len(args.shot) > 1:
            raise ValueError(f"finetuned models only supports loading one shot setting at a time. You are considering {len(args.shot)} different shot setting. shot: {args.shot}.")
        if len(args.prompt_type) > 1:
            raise ValueError(f"finetuned models only supports loading one prompt type at a time. You are considering {len(args.prompt_type)} different prompt type. prompt_type: {args.prompt_type}.")
    
    call_model = load_model(
        args.model, 
        device, 
        gen_mode=args.gen_mode,
        finetuned = args.finetuned_model,
        shot = args.shot[0],
        prompt_type = args.prompt_type[0],
        api_key = args.api_key,
        ft_mode = args.ft_mode,
        eval_task_theme = args.eval_task_theme,
    )

    low_semantic = True if args.low_semantic == 1 else False

    for shot in args.shot:
        for prompt_type in args.prompt_type:
            for task_id in args.task_id:
                if low_semantic:
                    shot = int(shot // 2)
                inference(
                    args.model,
                    call_model,
                    shot,
                    prompt_type,
                    task_id,
                    args.overwrite,
                    args.gen_mode,
                    args.finetuned_model,
                    args.data_mode,
                    args.ft_mode,
                    args.eval_task_theme,
                    low_semantic,
                )
