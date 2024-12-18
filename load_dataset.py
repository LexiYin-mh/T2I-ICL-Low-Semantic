import os, random
root_dir = os.path.dirname(os.path.abspath(__file__))

from helper import set_seed, read_json, find_image, find_caption
from configs import task_dataframe, instruction_dict, num_prompt_dict, item2word, low_semantic_mapping
from itertools import permutations

def load_inputs(
    shot,
    prompt_type,
    task_id,
    x_idxs,
    theta_idxs,
    x_list,
    theta_list,
    include_output = False,
):
    text_inputs, image_inputs = [], []
    theta = theta_list[theta_idxs[shot+1]]
    x_demos = []
    
    for demo_idx in range(shot+1):
        x_idx = x_list[x_idxs[demo_idx]]
        x_idx = item2word.get(x_idx, x_idx)
        x_demos.append(x_idx)
        
        if prompt_type == 'misleading':
            # misleading
            theta_idx = theta_list[theta_idxs[demo_idx]]
            text_inputs.append(f"{x_idx} {theta_idx}: ")
        elif prompt_type == 'exact':
            # exact
            text_inputs.append(f"{x_idx} {theta}: ")
        else:
            text_inputs.append(f"{x_idx}: ")
        num_images = shot + include_output
            
        if demo_idx < num_images:
            image_inputs.append(find_image(
                root_dir, 
                task_id, 
                x_idxs[demo_idx], 
                theta, 
            ))
    
    return {
        "text_inputs": text_inputs,
        "image_inputs": image_inputs,
        "x_list": x_demos,
        'theta': theta_list[theta_idxs[shot+1]],
        'x_idx': x_idxs[shot],
    }

def find_position_and_random_numbers(theta_list, theta, num_random=2, range_start=0, range_end=9):
    try:
        # 找到 theta 在列表中的位置
        position = theta_list.index(theta)
        
        # 生成随机数，确保它们不等于 position
        random_numbers = []
        while len(random_numbers) < num_random:
            rand_num = random.randint(range_start, range_end)
            if rand_num != position and rand_num not in random_numbers:
                random_numbers.append(rand_num)
        
        return position, random_numbers
    except ValueError:
        return None, "Theta not in theta_list"

def load_inputs_printing(
    i,
    shot,
    prompt_type,
    task_id,
    x_idxs,
    theta_idxs,
    x_list,
    theta_list,
    include_output = False,
):
    text_inputs, image_inputs = [], []
    theta = theta_list[theta_idxs[shot+1]]
    print(f'| theta: {theta}') # query object
    qury_color = None
    x_demos = []
    
    # iterate positive samples
    for demo_idx in range(shot+1):
        x_idx = x_list[x_idxs[demo_idx]]
        x_idx = item2word.get(x_idx, x_idx)
        x_demos.append(x_idx)
        
        # if prompt_type == 'misleading':
        #     # misleading
        #     theta_idx = theta_list[theta_idxs[demo_idx]]
        #     text_inputs.append(f"{x_idx} {theta_idx}: ")
        # elif prompt_type == 'exact':
        #     # exact
        #     text_inputs.append(f"{x_idx} {theta}: ")
        if demo_idx < shot:
            color_mapped = low_semantic_mapping[x_idx]
            theta_mapped = low_semantic_mapping[theta]
            text_inputs.append(f"{color_mapped} {theta_mapped}: ")
        # get the query color
        if demo_idx == shot:
            query_color = x_idx
            
        num_images = shot + include_output
            
        if demo_idx < num_images:
            image_inputs.append(find_image(
                root_dir, 
                task_id, 
                x_idxs[demo_idx], 
                theta, 
            ))
    
    # get extra shots for querycolor_objects
    # 获取theta在 theta_list的位置和random theta
    position, random_theta_ids = find_position_and_random_numbers(theta_list, theta, shot)
    for theta_id in random_theta_ids:
        random_theta = theta_list[theta_id]
        image_inputs.append(find_image(
            root_dir,
            task_id,
            x_idxs[shot],
            random_theta,
        ))
        color_mapped = low_semantic_mapping[x_list[x_idxs[shot]]]
        random_theta_mapped = low_semantic_mapping[random_theta]
        text_inputs.append(f"{color_mapped} {random_theta_mapped}: ")

    text_inputs.append(f"{color_mapped} {theta_mapped}: ")
        
    return {
        "text_inputs": text_inputs,
        "image_inputs": image_inputs,
        "x_list": x_demos,
        'theta': theta_list[theta_idxs[shot+1]],
        'x_idx': x_idxs[shot],
    }
    

def load_dataset(
    shot,
    prompt_type,
    task_id, 
    seed = 123,
    data_mode = 'default', # 'default' or 'ft_train' or 'ft_test'
    include_output = False, # whether include expected output in the prompt or not
    ft_mode = 'all', # 'all' or 'leave_one_out'
    low_semantic = False,
):
    print("========"*3)
    print(f'Loading the dataset for task {task_id}...')
    print(f'| task type: {task_dataframe[task_id]["task_type"]}')
    print(f'| x_space: {task_dataframe[task_id]["x_space"]}')
    print(f'| theta_space: {task_dataframe[task_id]["theta_space"]}')
    print(f'| prompt_type: {prompt_type}')
    print(f'| shot: {shot}')
    
    set_seed(seed)
    
    if data_mode == 'ft_train':
        prompts_list = read_json(f"{root_dir}/load_datasets/prompts_list_{data_mode}_{ft_mode}.json")
    else:
        prompts_list = read_json(f"{root_dir}/load_datasets/prompts_list_{data_mode}.json")
    
    if data_mode in ['default', 'ft_test'] or (data_mode == 'ft_train' and ft_mode == 'leave_one_out'):    
        # leave one out: ignore one task and use that as the evaluation set
        # in this case, give a task_id, we need to consider all data samples in this task
        data_loader = []
        for i in range(num_prompt_dict[data_mode]):
            item_inputs = prompts_list[i]
            if data_mode in ['default', 'ft_train']:
                theta_input = item_inputs["theta_list"]
            else:
                theta_input = [item_inputs["theta_list"][i%len(item_inputs["theta_list"])] for i in range(shot+2)]
            
            include_output = True if data_mode == 'ft_train' else False
            if low_semantic:
                input_dict = load_inputs_printing(
                    i,
                    shot,
                    prompt_type,
                    task_id,
                    item_inputs["x_list"],
                    theta_input,
                    task_dataframe[task_id]["x_list"],
                    task_dataframe[task_id]["theta_list"],
                    include_output = include_output,
                )
            else:
                input_dict = load_inputs(
                    shot,
                    prompt_type,
                    task_id,
                    item_inputs["x_list"],
                    theta_input,
                    task_dataframe[task_id]["x_list"],
                    task_dataframe[task_id]["theta_list"],
                    include_output = include_output,
                )
            if i < 10:
                print("======input dict ========")
                print(input_dict)
            input_dict['save_path'] = f"{i}_{input_dict['theta']}_{'_'.join(input_dict['x_list'])}"
            data_loader.append(input_dict)
    elif data_mode == 'ft_train':
        x_lists = list(permutations(prompts_list, shot + 1))
        theta_lists = list(permutations(prompts_list, 1))
        
        data_loader = []
        for x_list in x_lists:
            for theta_list in theta_lists:
                input_dict = load_inputs(
                    shot,
                    prompt_type,
                    task_id,
                    x_list,
                    [None for _ in range(shot+1)] + [theta_list[0]],
                    task_dataframe[task_id]["x_list"],
                    task_dataframe[task_id]["theta_list"],
                    include_output = include_output,
                )
                data_loader.append(input_dict)
    else:
        raise NotImplementedError(f"Unknown data_mode: {data_mode}!")
            
    print('Done!')
    print("========"*3)
    return data_loader
        
def get_instruction(
    prompt_type, 
    gen_mode,
    task_id, 
    model,
):
    if prompt_type == 'instruct':
        return (instruction_dict[prompt_type][gen_mode][task_id], '')
    elif prompt_type == 'caption':
        return (instruction_dict[prompt_type][gen_mode], '')
    elif prompt_type == 'cot':
        return instruction_dict[prompt_type][gen_mode]
    elif prompt_type in ['default', 'misleading', 'exact']:
        if model in instruction_dict['default'][gen_mode]:
            return instruction_dict['default'][gen_mode][model]
        else:
            raise NotImplementedError(f'{model} is not supported for {gen_mode} generation!')
    else:
        raise NotImplementedError(f'{prompt_type} is not supported!')
        
        
def get_prompt(
    text_inputs,
    image_inputs,
    prompt_type,
    task_id, 
    model,
    gen_mode, 
    history = None,
):
    if prompt_type in ['instruct', 'default', 'misleading', 'exact']: # [-1,0,1]:
        query = {
            'text_inputs': text_inputs, 
            'image_inputs': image_inputs,
            'instruction': get_instruction(
                prompt_type, 
                gen_mode,
                task_id,
                model,
            )
        }
    elif prompt_type == 'caption': # -2:
        for i, image_path in enumerate(image_inputs):
            caption = find_caption(image_path)
            text_inputs.insert(
                2*i+1, 
                caption + ' '
            )
            
        query = {
            'text_inputs': text_inputs,
            'image_inputs': [],
            'instruction': get_instruction(
                prompt_type, 
                gen_mode,
                task_id,
                model,
            ),
            'call_mode': 'text',
        }
    elif prompt_type == 'cot':
        if gen_mode == 'general':
            query = {
                'text_inputs': text_inputs, 
                'image_inputs': image_inputs,
                'instruction': get_instruction(
                    prompt_type, 
                    gen_mode,
                    task_id,
                    model,
                ),
                'save_history': True,
            }
        else:
            instruction = get_instruction(
                prompt_type, 
                gen_mode,
                task_id,
                model,
            )
            
            query = {
                'text_inputs': text_inputs, 
                'image_inputs': image_inputs,
                'instruction': instruction,
                'history': history,
            }
            
    else:
        raise NotImplementedError(f"Unknown prompt_type: {prompt_type}!")
    return query 