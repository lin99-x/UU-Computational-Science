#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct tree_node
{
   int              ID;
   char             *name;
   struct tree_node *left;
   struct tree_node *right;
} node_t;


void print_bst(node_t *node)
{
   if(node == NULL) {printf("Tree is empty!\n"); return;}

   if (node != NULL) printf("%d %s: \t", node->ID, node->name);
   if (node->left != NULL) printf("L%d,", node->left->ID);
   if (node->right != NULL) printf("R%d", node->right->ID);
   printf("\n");

   if (node->left != NULL)
      print_bst(node->left);
   if (node->right != NULL)
      print_bst(node->right);
}


void delete_tree(node_t **node)
{
  if (*node) {
   delete_tree(&(*node)->left);
   delete_tree(&(*node)->right);
   free(*node);
  }
}

void insert(node_t **node, int ID, char *name)
{
  node_t *tmp = NULL;
  char *target = strdup(name);
  if ((*node) == NULL) {
   tmp = (node_t *)malloc(sizeof(node_t));
   tmp->ID = ID;
   tmp->name = target;
   tmp->left = tmp->right = NULL;
   *node = tmp;
   return;
  }
  if (ID > (*node)->ID) {
   insert(&(*node)->right, ID, name);
  }
  else if (ID < (*node)->ID) {
   insert(&(*node)->left, ID, name);
  }
}

// question here
void search(node_t **node, int id)
{
   printf("@I am looking at node %d", (*node)->ID);
   if ((*node) == NULL) {
   printf("Plant with ID %d does not exist!", id);
  }
  if ((*node)->ID == id) {
   printf("Plant with ID %d has name %s", id, (*node)->name);
   }
   else if ((*node)->ID < id) {
      search(&(*node)->right, id);
   }
   else if ((*node)->ID > id) {
      search(&(*node)->left, id);
   }
}



int main(int argc, char const *argv[])
{
   node_t *root = NULL;  // empty tree; *root is pointer to root node
   printf("Inserting nodes to the binary tree.\n");
   insert(&root, 445, "sequoia");

   insert(&root, 162, "fir");
   insert(&root, 612, "baobab");
   insert(&root, 845, "spruce");
   insert(&root, 862, "rose");
   insert(&root, 168, "banana");
   insert(&root, 225, "orchid");
   insert(&root, 582, "chamomile");  
   printf("Printing nodes of the tree.\n");
   
   print_bst(root);
   printf("hello world");
   printf("hi");
   search(&root, 445);
   search(&root, 467);

   printf("Deleting tree.\n");
   delete_tree(&root);

   print_bst(root);
   return 0;
}
