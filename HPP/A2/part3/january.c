#include <stdio.h>
#include <stdlib.h>


/* define a linked list node. */
typedef struct node
{
    int day;
    float max;
    float min;
    struct node * next;
}node_t;


void print_list(node_t * head) {
    // node_t * current = head;
    // head is actually node
    if (head == NULL) {printf("Database is empty!\n"); return;}
    while (head != NULL) {
        printf("%2d    %10f   %10f\n", head->day, head->min, head->max);
        head = head->next;
    }
}


void delete_by_index(node_t ** head, int id) {
    node_t * tmp = *head;
    node_t * pre;

    if (tmp != NULL && tmp->day == id) {
        *head = tmp->next;
        free(tmp);
        return;
    }

    while (tmp != NULL && tmp->day != id) {
        pre = tmp;
        tmp = tmp->next;
    }

    if (tmp == NULL) return;

    pre->next = tmp->next;
    free(tmp);
    // free(pre);
}


void insert(node_t ** head, int id, double min, double max) {
    node_t * tmp = NULL;
    if (*head == NULL) {
        tmp = (node_t *)malloc(sizeof(node_t));
        tmp->day = id;
        tmp->max = max;
        tmp->min = min;
        tmp->next = NULL;
        *head = tmp;
        return;
    }
    if (id > (*head)->day) {
        insert(&(*head)->next, id, min, max);
    }
    else if (id < (*head)->day) {
        tmp = (node_t *)malloc(sizeof(node_t));
        tmp->day = id;
        tmp->max = max;
        tmp->min = min;
        tmp->next = *head;
        *head = tmp;

    } else if (id == (*head)->day){
    	(*head)->max = max;
    	(*head)->min = min;
    }
}


int main() {
    char index;
    int day, i;
    float a, b, min, max; 
    node_t * root = NULL;
    for (;;) {
        printf("Enter command: ");
        scanf(" %c", &index); // commands starting with whitespaces are considered as valid
        if (index == 'A') {
            i = scanf("%d", &day);
			if (i==0|| day<1 || day>31){
				printf("You entered an invalid day. Start over.\n");
				while (getchar() != '\n');
				continue;
			}
			i = scanf("%f", &a);
			if (i==0){
				printf("You entered an invalid min. Start over.\n");
				while (getchar() != '\n');
				continue;
			}
			
			i = scanf("%f", &b);
			if (i==0){
				printf("You entered an invalid max. Start over.\n");
				while (getchar() != '\n');
				continue;
			}
            // note that though there are several scanf, user can give all data (with in a command) in one line 
            
            if (a < b) {
                min = a;
                max = b;
            }
            else {
                min = b;
                max = a;
            }
            			
            while (getchar() != '\n'); 
            // we assume at most one command in one line, thus flush all remaining buffer
            // in this sense, "A 1 -15.2 -5.1 hahaha" can be seen as a valid command
            // Note: commands like "A 1.5 10" would work as if it were "A 1 0.5 10" 
            
            insert(&root, day, min, max);
        }
        else if (index == 'D') {
            // remove the day with the given index from the database
            // if user trys to remove an index that does not exist, we don't print an error message
            i = scanf("%d", &day);
			if (i==0|| day<1 || day>31){
				printf("You entered an invalid day. Start over.\n");
				while (getchar() != '\n');
				continue;
            }
            while (getchar() != '\n');
            delete_by_index(&root, day);
        }
        else if (index == 'P') {
            // print all data as a table
            while (getchar() != '\n');
            printf("day    min	    max\n");
            print_list(root);
        }
        else if (index == 'Q') {
            break;
        }
        else if (index == '\n'){
            continue;
        }
        else {
            printf("Please enter a valid command.\n");
			while (getchar() != '\n');
			continue;
        }
    }
	// free(root);
    return 0;
}
