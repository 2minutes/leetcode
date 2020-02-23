using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[AddComponentMenu("MyGame/Ball")]
public class Ball : MonoBehaviour
{
	public float m_speed=10;
	// public float m_liveTime=30;
	public float m_power=1.0f;
	protected Transform m_transform;
    private Rigidbody player_rg;
    private Rigidbody ball_rg;
    private bool isPickUp;
    private Vector3 carry_offset;
    // public float shot_speed;


    bool canShoot=false;
    // Start is called before the first frame update
    void Start()
    {
        isPickUp = false;
        carry_offset = new Vector3(0.5f, 1.0f, 0.5f);
        m_transform=this.transform;
        // button.onClick.AddListener(TaskOnClick);
        ball_rg = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        // m_liveTime-=Time.deltaTime;
        // if(m_liveTime<=0){
        // 	Destroy(this.gameObject);
        // }
        
        if (isPickUp)
        {
            if(!canShoot){
                if(player_rg!=null){
                    transform.position = player_rg.position + carry_offset;
                }
            }else{
                m_transform.Translate(new Vector3(0,0,m_speed*Time.deltaTime));
            }
            if(Input.GetKey(KeyCode.Space)||Input.GetMouseButton(0)){
                canShoot=true;
            }
        }
    }

    void OnTriggerEnter(Collider other){
        if(other.tag.CompareTo("Player")==0){
            player_rg = other.GetComponent<Collider>().attachedRigidbody;
            isPickUp = true;
        }else if(other.tag.CompareTo("Enemy")==0||other.tag.CompareTo("Enemy2")==0){
            Destroy(this.gameObject);
        }
        // Destroy(this.gameObject);
    }

    // void TaskOnClick()
    // {
    //     if (isPickUp)
    //     {
            // transform.position = new Vector3(transform.position.x, 0.5f, transform.position.z);
            // ball_rg.velocity = player_rg.transform.forward * shot_speed;
            // isPickUp = false;
    //     }
    // }
}
