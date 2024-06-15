import os
import re

def convert_markdown_images(directory):
    # Regex pattern to match [![](image_url)](link_url) format
    img_link_regex = re.compile(r'\[!\[\]\((.*?)\)\]\((.*?)\)')

    # Walk through all directories and subdirectories
    for root, _, files in os.walk(directory):
        for filename in files:
            # Process only .md files
            if filename.endswith(".md"):
                filepath = os.path.join(root, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()

                def replace_img_link(match):
                    full_match = match.group(0)
                    image_url = match.group(1)
                    link_url = match.group(2)

                    # Construct the HTML img tag with optional link and width/height attributes
                    if link_url.strip():
                        img_tag = f'<a href="{link_url}"><img src="{image_url}" alt="Image" width="80%" height="auto"></a>'
                    else:
                        img_tag = f'<img src="{image_url}" alt="Image" width="80%" height="auto">'

                    return img_tag

                # Replace [![]()]() with <img> format including width/height attributes
                new_content = img_link_regex.sub(replace_img_link, content)

                # Write the updated content back to the file if changes were made
                if new_content != content:
                    with open(filepath, 'w', encoding='utf-8') as file:
                        file.write(new_content)
                    print(f"Updated {filepath}")

if __name__ == "__main__":
    directory = '.'  # Specify the directory containing the .md files
    convert_markdown_images(directory)
